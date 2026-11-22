"""Modul 3 ReSnap versi lokal: pencocokan resep tanpa API atau LLM.

Database: data/recipes.json.gz (dibangun oleh scripts/build_recipe_db.py dari
14 ribu resep Cookpad Indonesia). Kontrak sama dengan recipe_fallback:

    recommend(ingredients: list[str], top_k: int = 5) -> list[dict]

Skor sesuai PRD:  skor(r) = |B ∩ R| / |R|
  B = bahan terdeteksi (nama kelas YOLO)
  R = bahan utama resep tanpa bumbu dasar; bahan yang tidak bisa dideteksi
      (mis. daging sapi, santan, tepung) tetap masuk R sebagai bahan kurang.
Seri -> bahan kurang paling sedikit, lalu jumlah bahan cocok terbanyak, lalu resep terpopuler.
Filter relevansi: resep yang memakai >= 2 bahan terdeteksi ditampilkan lebih dulu.
"""
from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

from .detector import display_name

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "recipes.json.gz"
MIN_MAIN_INGREDIENTS = 2   # resep yang bahan utamanya cuma satu (mis. "telur rebus") tidak ditampilkan


def _load() -> list[dict]:
    return _load_path(str(DB_PATH))


@lru_cache(maxsize=2)
def _load_path(path: str) -> list[dict]:
    raw = Path(path).read_bytes()
    data = json.loads(gzip.decompress(raw) if path.endswith(".gz") else raw)
    recipes = data["resep"]
    for r in recipes:
        r["_set"] = frozenset(r["bahan_kelas"])
        r["_total"] = len(r["bahan_kelas"]) + len(r["bahan_lain"])
        r["_key"] = re.sub(r"[^a-z]", "", r["judul"].lower())
    return recipes


def available() -> bool:
    return DB_PATH.is_file()


def score(recipe: dict, detected: set[str]) -> tuple[float, list[str], list[str]]:
    match = [c for c in recipe["bahan_kelas"] if c in detected]
    missing = [display_name(c) for c in recipe["bahan_kelas"] if c not in detected] + recipe["bahan_lain"]
    return len(match) / recipe["_total"], [display_name(c) for c in match], missing


def _rank(detected: set[str], min_match: int) -> list[tuple]:
    ranked = []
    for r in _load():
        if r["_total"] < MIN_MAIN_INGREDIENTS or len(r["_set"] & detected) < min_match:
            continue
        s, have, missing = score(r, detected)
        ranked.append(((-s, len(missing), -len(have), -r["loves"], r["id"]), r, s, have, missing))
    ranked.sort(key=lambda x: x[0])
    return ranked


def recommend(ingredients: list[str], top_k: int = 5) -> list[dict]:
    """Top-k resep. Utamakan resep yang memakai minimal 2 bahan terdeteksi (bila B punya >= 2 bahan),
    lalu isi sisanya dengan resep yang memakai 1 bahan. Urutan dalam tiap kelompok mengikuti skor PRD."""
    detected = {i for i in ingredients if i}
    if not detected:
        return []
    tiers = [min(2, len(detected))] + ([1] if len(detected) >= 2 else [])

    out, seen_titles, seen_ids = [], set(), set()
    for min_match in tiers:
        for _, r, s, have, missing in _rank(detected, min_match):
            if r["_key"] in seen_titles or r["id"] in seen_ids:   # judul kembar di Cookpad
                continue
            seen_titles.add(r["_key"])
            seen_ids.add(r["id"])
            out.append({
                "judul": r["judul"], "gambar": None, "skor": s,
                "bahan_tersedia": have, "bahan_kurang": missing,
                "bahan_resep": r["bahan_asli"], "langkah": r["langkah"],
                "sumber": r["sumber"], "kategori": r["kategori"], "loves": r["loves"], "api": "lokal",
            })
            if len(out) == top_k:
                return out
    return out
