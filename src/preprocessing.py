"""Modul 1 ReSnap: pra-pemrosesan citra tradisional (OpenCV).

Urutan tahap sesuai PRD 3.1:
    1. Resize + konversi ruang warna (BGR -> RGB / HSV)
    2. Reduksi noise (Gaussian Blur 5x5 + Median Blur)
    3. Perbaikan kontras (CLAHE pada kanal L di ruang LAB)
    4. Segmentasi Otsu (grayscale -> mask biner)
    5. Morfologi (opening/closing) + findContours
    6. Letterbox 640x640 + normalisasi piksel [0, 255] -> [0.0, 1.0]

Catatan desain:
- Resize mempertahankan rasio aspek (sisi terpanjang = 640) lalu diberi
  letterbox, karena YOLOv8 dilatih dengan letterbox. Resize paksa 640x640
  akan membuat bahan terlihat gepeng dan menurunkan akurasi.
- Model v2/v3 dilatih dengan foto mentah. Karena itu, foto yang masuk ke
  YOLO secara default hanya di-letterbox dan dinormalisasi. Denoise + CLAHE
  bisa diaktifkan lewat `enhance_for_detection=True`; pengaruhnya diukur
  dengan scripts/evaluate_export.py --ablation.
- Mask Otsu dan kontur dipakai untuk analisis/visualisasi (kandidat area
  bahan), bukan untuk memotong input YOLO.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

TARGET_SIZE = 640
PAD_VALUE = 114  # nilai abu-abu padding yang sama dengan Ultralytics


@dataclass
class LetterboxInfo:
    """Parameter untuk memetakan koordinat 640x640 kembali ke foto asli."""
    scale: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int

    def to_original(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
        if boxes.size == 0:
            return boxes.reshape(0, 4)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - self.pad_x) / self.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - self.pad_y) / self.scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.orig_h)
        return boxes


@dataclass
class PreprocessResult:
    """Semua keluaran antara, supaya UI dan laporan bisa menampilkan tiap tahap."""
    original_bgr: np.ndarray
    resized_bgr: np.ndarray
    rgb: np.ndarray
    hsv: np.ndarray
    denoised_bgr: np.ndarray
    clahe_bgr: np.ndarray
    gray: np.ndarray
    otsu_threshold: float
    otsu_mask: np.ndarray
    morph_mask: np.ndarray
    contours: list
    contour_overlay_bgr: np.ndarray
    letterboxed_bgr: np.ndarray
    letterbox: LetterboxInfo
    model_input: np.ndarray  # float32 RGB, HxWxC, rentang [0, 1]
    stats: dict = field(default_factory=dict)

    def stages(self) -> dict[str, np.ndarray]:
        """Gambar per tahap dalam RGB/grayscale uint8 untuk ditampilkan."""
        return {
            "1. Resize (RGB)": self.rgb,
            "1b. Kanal Hue (HSV)": (self.hsv[:, :, 0].astype(np.float32) * 255 / 179).astype(np.uint8),
            "2. Gaussian 5x5 + Median": bgr_to_rgb(self.denoised_bgr),
            "3. CLAHE kanal L (LAB)": bgr_to_rgb(self.clahe_bgr),
            "4. Otsu (mask biner)": self.otsu_mask,
            "5. Morfologi + kontur": bgr_to_rgb(self.contour_overlay_bgr),
            "6. Letterbox 640x640": bgr_to_rgb(self.letterboxed_bgr),
        }


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_image(data: bytes) -> np.ndarray:
    """Bytes dari st.camera_input / st.file_uploader -> BGR uint8."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File bukan gambar yang bisa dibaca.")
    return img


# 1. Resize & color space -----------------------------------------------------

def resize_keep_ratio(
    img: np.ndarray, target: int = TARGET_SIZE, interpolation: int = cv2.INTER_LINEAR
) -> tuple[np.ndarray, float]:
    """INTER_LINEAR = interpolasi yang dipakai Ultralytics saat training, jadi input konsisten."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation), scale


def to_color_spaces(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGB untuk tampilan/model, HSV untuk memisahkan warna dari intensitas cahaya."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


# 2. Noise reduction ----------------------------------------------------------

def reduce_noise(bgr: np.ndarray, gaussian_ksize: int = 5, median_ksize: int = 3) -> np.ndarray:
    """Gaussian meredam noise sensor (Gaussian), median membuang bintik salt-and-pepper."""
    out = cv2.GaussianBlur(bgr, (gaussian_ksize, gaussian_ksize), 0)
    return cv2.medianBlur(out, median_ksize)


# 3. Contrast -----------------------------------------------------------------

def apply_clahe(bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """CLAHE hanya di kanal L supaya warna bahan tidak bergeser."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab = cv2.merge((clahe.apply(l_ch), a_ch, b_ch))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# 4. Otsu segmentation --------------------------------------------------------

def otsu_segment(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Grayscale -> mask biner Otsu.

    Polaritas dipilih otomatis: latar (talenan/meja) diasumsikan mendominasi
    tepi foto, jadi warna mayoritas di tepi dijadikan latar (0).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)
    return gray, mask, float(thresh)


# 5. Morphology & contours ----------------------------------------------------

def clean_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 2) -> np.ndarray:
    """Opening membuang bintik kecil, closing menutup lubang di dalam objek."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def find_candidate_contours(mask: np.ndarray, min_area_ratio: float = 0.002) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * mask.shape[0] * mask.shape[1]
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]
    return sorted(kept, key=cv2.contourArea, reverse=True)


def draw_contours(bgr: np.ndarray, contours: list) -> np.ndarray:
    out = bgr.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 160, 0), 1)
    return out


# 6. Letterbox & normalization ------------------------------------------------

def letterbox(img: np.ndarray, target: int = TARGET_SIZE) -> tuple[np.ndarray, LetterboxInfo]:
    h, w = img.shape[:2]
    resized, scale = resize_keep_ratio(img, target)
    rh, rw = resized.shape[:2]
    pad_x, pad_y = (target - rw) // 2, (target - rh) // 2
    canvas = np.full((target, target, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + rh, pad_x:pad_x + rw] = resized
    return canvas, LetterboxInfo(scale, pad_x, pad_y, w, h)


def normalize(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float32 [0, 1]."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# Pipeline --------------------------------------------------------------------

def enhance(bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Denoise + CLAHE pada resolusi asli (dipakai bila enhance_for_detection aktif)."""
    return apply_clahe(reduce_noise(bgr), clip_limit=clip_limit)


def run_pipeline(
    bgr: np.ndarray,
    enhance_for_detection: bool = False,
    clahe_clip: float = 2.0,
    morph_kernel: int = 5,
) -> PreprocessResult:
    if bgr is None or bgr.ndim != 3:
        raise ValueError("Input harus citra BGR 3 kanal.")

    resized, _ = resize_keep_ratio(bgr, interpolation=cv2.INTER_AREA)  # tampilan lebih halus
    rgb, hsv = to_color_spaces(resized)
    denoised = reduce_noise(resized)
    clahe = apply_clahe(denoised, clip_limit=clahe_clip)
    gray, otsu_mask, otsu_t = otsu_segment(clahe)
    morph = clean_mask(otsu_mask, kernel_size=morph_kernel)
    contours = find_candidate_contours(morph)
    overlay = draw_contours(clahe, contours)

    source = enhance(bgr, clahe_clip) if enhance_for_detection else bgr
    boxed, info = letterbox(source)
    model_input = normalize(boxed)

    stats = {
        "ukuran_asli": f"{bgr.shape[1]}x{bgr.shape[0]}",
        "ambang_otsu": round(otsu_t, 1),
        "kandidat_kontur": len(contours),
        "rasio_foreground": round(float((morph > 0).mean()), 3),
        "kecerahan_rata2_L_sebelum": round(float(cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)[:, :, 0].mean()), 1),
        "kecerahan_rata2_L_sesudah": round(float(cv2.cvtColor(clahe, cv2.COLOR_BGR2LAB)[:, :, 0].mean()), 1),
        "enhance_untuk_deteksi": enhance_for_detection,
    }
    return PreprocessResult(
        original_bgr=bgr, resized_bgr=resized, rgb=rgb, hsv=hsv,
        denoised_bgr=denoised, clahe_bgr=clahe, gray=gray,
        otsu_threshold=otsu_t, otsu_mask=otsu_mask, morph_mask=morph,
        contours=contours, contour_overlay_bgr=overlay,
        letterboxed_bgr=boxed, letterbox=info, model_input=model_input, stats=stats,
    )
