"""ReSnap: deteksi bahan + rekomendasi resep dalam satu panggilan (untuk integrasi frontend).

Pemakaian sebagai modul:
    from resnap_pipeline import scan, rekomendasi
    hasil = scan("foto.jpg")                       # dict siap jadi JSON
    hasil2 = rekomendasi(["egg", "tomato"])        # setelah user mengedit daftar bahan

Pemakaian dari terminal:
    python resnap_pipeline.py foto.jpg --out hasil.json --gambar hasil.jpg

Contoh keluaran tersedia di contoh/contoh_output.json.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.detector import DEFAULT_WEIGHTS, INGREDIENTS, IngredientDetector, display_name, draw_detections
from src.preprocessing import decode_image, run_pipeline
from src.recipe_local import recommend as _recommend

VERSI_MODEL = "v4b (YOLOv8s, 30 kelas)"


@lru_cache(maxsize=1)
def _detector(weights: str = str(DEFAULT_WEIGHTS)) -> IngredientDetector:
    return IngredientDetector(weights)


def _read(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (bytes, bytearray)):
        return decode_image(bytes(image))
    img = cv2.imread(str(image))
    if img is None:
        raise ValueError(f"Gambar tidak bisa dibaca: {image}")
    return img


def rekomendasi(bahan: list[str], top_k: int = 5) -> list[dict]:
    """Rekomendasi resep dari daftar kelas bahan (mis. hasil edit user di frontend)."""
    tidak_dikenal = sorted(set(bahan) - set(INGREDIENTS))
    if tidak_dikenal:
        raise ValueError(f"Kelas bahan tidak dikenal: {tidak_dikenal}. Pilihan: {list(INGREDIENTS)}")
    return _recommend(bahan, top_k=top_k)


def scan(image, conf: float = 0.5, iou: float = 0.7, top_k: int = 5,
         enhance: bool = False, simpan_gambar: str | None = None) -> dict:
    """Foto (path, bytes, atau array BGR) -> deteksi + daftar bahan + Top-k resep."""
    t0 = time.perf_counter()
    bgr = _read(image)
    pre = run_pipeline(bgr, enhance_for_detection=enhance)
    t1 = time.perf_counter()
    det = _detector().detect(pre, conf=conf, iou=iou)
    t2 = time.perf_counter()
    bahan_kelas = sorted(det.ingredients, key=lambda c: -det.ingredients[c]["conf_maks"])
    resep = _recommend(bahan_kelas, top_k=top_k) if bahan_kelas else []
    t3 = time.perf_counter()

    if simpan_gambar:
        cv2.imwrite(simpan_gambar, cv2.cvtColor(draw_detections(bgr, det), cv2.COLOR_RGB2BGR))

    return {
        "gambar": {"lebar": int(bgr.shape[1]), "tinggi": int(bgr.shape[0])},
        "deteksi": [
            {"kelas": d.cls, "nama": display_name(d.cls), "confidence": round(d.conf, 4),
             "box": [round(v, 1) for v in d.box]}
            for d in det.detections
        ],
        "bahan": [
            {"kelas": c, "nama": display_name(c), "jumlah": det.ingredients[c]["jumlah"],
             "confidence_maks": round(det.ingredients[c]["conf_maks"], 4)}
            for c in bahan_kelas
        ],
        "resep": [
            {k: r[k] for k in ("judul", "skor", "bahan_tersedia", "bahan_kurang", "bahan_resep",
                               "langkah", "kategori", "loves", "sumber")}
            for r in resep
        ],
        "pengaturan": {"conf": conf, "iou": iou, "top_k": top_k, "enhance": enhance, "model": VERSI_MODEL},
        "waktu_ms": {"preprocessing": round(1000 * (t1 - t0)), "deteksi": round(1000 * (t2 - t1)),
                     "resep": round(1000 * (t3 - t2))},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("foto")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", help="simpan JSON ke file")
    ap.add_argument("--gambar", help="simpan foto dengan bounding box")
    args = ap.parse_args()
    hasil = scan(args.foto, conf=args.conf, top_k=args.top_k, simpan_gambar=args.gambar)
    teks = json.dumps(hasil, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(teks, encoding="utf-8")
    print(teks)


if __name__ == "__main__":
    main()
