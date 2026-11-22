"""Modul 2 ReSnap: inference YOLOv8 + post-processing (Affan).

Alur: hasil letterbox dari preprocessing -> YOLOv8 (NMS bawaan Ultralytics)
-> filter confidence -> koordinat dikembalikan ke foto asli -> himpunan bahan unik B.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import PreprocessResult

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "best.pt"

# Nama tampilan Bahasa Indonesia + nama untuk query Recipe API (Inggris).
INGREDIENTS = {
    "tomato": ("Tomat", "tomato"),
    "egg": ("Telur", "egg"),
    "onion": ("Bawang bombay", "onion"),
    "shallot": ("Bawang merah", "shallot"),
    "garlic": ("Bawang putih", "garlic"),
    "chili": ("Cabai", "chili pepper"),
    "carrot": ("Wortel", "carrot"),
    "potato": ("Kentang", "potato"),
    "cabbage": ("Kubis", "cabbage"),
    "broccoli": ("Brokoli", "broccoli"),
    "cauliflower": ("Kembang kol", "cauliflower"),
    "cucumber": ("Timun", "cucumber"),
    "eggplant": ("Terong", "eggplant"),
    "mushroom": ("Jamur", "mushroom"),
    "spinach": ("Bayam", "spinach"),
    "green_onion": ("Daun bawang", "green onion"),
    "green_beans": ("Buncis", "green beans"),
    "long_beans": ("Kacang panjang", "long beans"),
    "corn": ("Jagung", "corn"),
    "pumpkin": ("Labu", "pumpkin"),
    "lime": ("Jeruk nipis", "lime"),
    "ginger": ("Jahe", "ginger"),
    "turmeric": ("Kunyit", "turmeric"),
    "galangal": ("Lengkuas", "galangal"),
    "lemongrass": ("Serai", "lemongrass"),
    "tofu": ("Tahu", "tofu"),
    "tempeh": ("Tempe", "tempeh"),
    "chicken": ("Ayam", "chicken"),
    "fish": ("Ikan", "fish"),
    "shrimp": ("Udang", "shrimp"),
}


def display_name(cls: str) -> str:
    return INGREDIENTS.get(cls, (cls.replace("_", " ").title(), cls))[0]


def query_name(cls: str) -> str:
    return INGREDIENTS.get(cls, (cls, cls.replace("_", " ")))[1]


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # xyxy pada foto asli


@dataclass
class DetectionResult:
    detections: list[Detection]
    ingredients: dict[str, dict]  # cls -> {"jumlah", "conf_maks"}

    @property
    def ingredient_set(self) -> set[str]:
        return set(self.ingredients)


class IngredientDetector:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "cpu"):
        from ultralytics import YOLO  # impor lambat agar modul lain tetap ringan

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Bobot model tidak ditemukan: {weights}")
        self.model = YOLO(str(weights), task="detect")
        self.device = device
        names = self.model.names
        self.names = names if isinstance(names, dict) else dict(enumerate(names))

    def detect(
        self,
        pre: PreprocessResult,
        conf: float = 0.5,
        iou: float = 0.7,
        agnostic_nms: bool = False,
    ) -> DetectionResult:
        import torch

        # Input sudah RGB float [0,1] 640x640 dari preprocessing -> tensor BCHW.
        tensor = torch.from_numpy(pre.model_input).permute(2, 0, 1).unsqueeze(0).contiguous()
        res = self.model.predict(
            tensor, conf=conf, iou=iou, imgsz=640, device=self.device,
            agnostic_nms=agnostic_nms, verbose=False,
        )[0]

        boxes = res.boxes
        xyxy = pre.letterbox.to_original(boxes.xyxy.cpu().numpy()) if len(boxes) else np.zeros((0, 4))
        confs = boxes.conf.cpu().numpy() if len(boxes) else []
        clss = boxes.cls.cpu().numpy().astype(int) if len(boxes) else []

        detections = [
            Detection(self.names[int(c)], float(p), tuple(map(float, b)))
            for b, p, c in zip(xyxy, confs, clss)
        ]
        detections.sort(key=lambda d: d.conf, reverse=True)

        ingredients: dict[str, dict] = {}
        for d in detections:
            item = ingredients.setdefault(d.cls, {"jumlah": 0, "conf_maks": 0.0})
            item["jumlah"] += 1
            item["conf_maks"] = max(item["conf_maks"], d.conf)
        return DetectionResult(detections, ingredients)


def _color(cls_name: str) -> tuple[int, int, int]:
    rng = np.random.default_rng(abs(hash(cls_name)) % (2**32))
    return tuple(int(v) for v in rng.integers(60, 230, size=3))


def draw_detections(bgr: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Gambar bounding box + label di foto asli. Mengembalikan RGB."""
    out = bgr.copy()
    thick = max(2, round(max(out.shape[:2]) / 400))
    font_scale = max(0.5, max(out.shape[:2]) / 1400)
    for d in result.detections:
        x1, y1, x2, y2 = map(int, d.box)
        color = _color(d.cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"{display_name(d.cls)} {d.conf:.0%}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        y_text = max(th + base, y1)
        cv2.rectangle(out, (x1, y_text - th - base), (x1 + tw, y_text), color, -1)
        cv2.putText(out, label, (x1, y_text - base), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
