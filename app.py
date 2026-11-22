"""Antarmuka Streamlit untuk deteksi bahan dan rekomendasi resep Smart Recipe."""

import hashlib
import logging
from base64 import b64encode
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = ASSETS / "logo.png"

st.set_page_config(
    page_title="Smart Recipe",
    page_icon=str(LOGO) if LOGO.is_file() else "🥬",
    layout="wide",
)

try:
    import cv2

    from resnap_pipeline import rekomendasi, scan
    from src.detector import INGREDIENTS, Detection, DetectionResult, display_name, draw_detections
    from src.preprocessing import decode_image
except (ImportError, OSError):
    logging.exception("Gagal mengimpor dependensi Smart Recipe")
    st.error("Dependensi aplikasi belum siap. Periksa instalasi paket sesuai petunjuk di README.")
    st.stop()


MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_PHOTO_PIXELS = 24_000_000


def read_photo(upload):
    """Simpan foto dan hasilnya hanya untuk unggahan yang sedang aktif."""
    if upload is None:
        photo_id = None
    elif upload.size > MAX_PHOTO_BYTES:
        photo_id = "too_large"
    else:
        data = upload.getvalue()
        photo_id = hashlib.sha256(data).hexdigest()

    state = st.session_state
    if photo_id != state.get("photo_id"):
        for key in ("photo", "photo_error", "scan_result", "annotated_photo", "ingredients", "recipes"):
            state.pop(key, None)
        state.photo_id = photo_id
        if photo_id == "too_large":
            state.photo_error = "Ukuran foto melebihi 10 MB. Pilih foto yang lebih kecil."
        elif photo_id is not None:
            try:
                photo = decode_image(data)
            except (ValueError, cv2.error):
                state.photo_error = "Foto tidak bisa dibaca. Coba unggah ulang sebagai JPG atau PNG."
            else:
                if photo.shape[0] * photo.shape[1] > MAX_PHOTO_PIXELS:
                    state.photo_error = "Resolusi foto melebihi 24 megapiksel. Perkecil foto lalu coba lagi."
                else:
                    state.photo = photo

    if state.get("photo_error"):
        st.warning(state.photo_error)
    return state.get("photo")


def scan_photo(photo):
    state = st.session_state
    with st.spinner("Mendeteksi bahan dan mencari resep…"):
        try:
            result = scan(photo)
            boxes = [
                Detection(item["kelas"], item["confidence"], tuple(item["box"]))
                for item in result["deteksi"]
            ]
            annotated = draw_detections(photo, DetectionResult(boxes, {})) if boxes else None
        except (ImportError, OSError):
            logging.exception("Model atau dataset gagal dimuat")
            st.error("Model atau dataset gagal dimuat. Periksa dependensi dan lokasi berkas proyek.")
            return
        except Exception:
            logging.exception("Pemindaian foto gagal")
            st.error("Pemindaian gagal. Coba pindai ulang atau pilih bahan secara manual.")
            return

    selected = [item["kelas"] for item in result["bahan"] if item["kelas"] in INGREDIENTS]
    state.scan_result = result
    state.annotated_photo = annotated
    state.ingredients = selected
    if len(selected) == len(result["bahan"]):
        state.recipes = (tuple(sorted(selected)), result["resep"])
    else:
        state.pop("recipes", None)


def show_detection(photo):
    state = st.session_state
    annotated = state.get("annotated_photo")
    if annotated is None:
        st.image(photo, channels="BGR", width="stretch", caption="Foto bahan")
    else:
        st.image(annotated, width="stretch", caption="Kotak dan confidence hasil deteksi")

    if "scan_result" not in state:
        return
    detected = state.scan_result["bahan"]
    if not detected:
        st.info("Belum ada bahan yang terdeteksi. Kamu bisa memilih bahan secara manual di bawah.")
        return
    with st.expander(f"Detail deteksi · {len(detected)} jenis bahan", expanded=True):
        for item in detected:
            st.write(
                f"**{item['nama']}** · {item['jumlah']} objek terdeteksi"
                f" · confidence tertinggi {item['confidence_maks']:.0%}"
            )


def show_recipes(selected):
    state = st.session_state
    if not selected:
        state.pop("recipes", None)
        st.info("Pilih setidaknya satu bahan untuk melihat rekomendasi.")
        return

    selection = tuple(sorted(selected))
    cached = state.get("recipes")
    if cached is None or cached[0] != selection:
        try:
            with st.spinner("Mencari resep yang cocok…"):
                cached = (selection, rekomendasi(list(selection)))
        except Exception:
            logging.exception("Pencarian resep gagal")
            st.error("Resep gagal dimuat. Periksa dataset lalu coba ubah pilihan bahan.")
            return
        state.recipes = cached

    recipes = cached[1]
    if not recipes:
        st.info("Belum ada resep yang cocok. Coba tambahkan bahan lain.")
        return

    for index, recipe in enumerate(recipes):
        with st.container(border=True, key=f"recipe_card_{index}"):
            st.markdown("### " + recipe["judul"])
            st.progress(recipe["skor"], text=f"{recipe['skor']:.0%} kecocokan bahan utama")
            if recipe["bahan_tersedia"]:
                st.write("**Tersedia** · " + ", ".join(recipe["bahan_tersedia"]))
            if recipe["bahan_kurang"]:
                st.write("**Perlu disiapkan** · " + ", ".join(recipe["bahan_kurang"]))
            with st.expander("Bahan lengkap dan cara memasak"):
                st.markdown("**Daftar bahan**")
                for ingredient in recipe["bahan_resep"]:
                    st.write("• " + ingredient)
                st.markdown("**Langkah memasak**")
                for number, step in enumerate(recipe["langkah"], 1):
                    st.write(f"{number}. {step}")
                if recipe["sumber"]:
                    st.link_button("Buka resep sumber", recipe["sumber"])


@st.cache_data(show_spinner=False)
def hero_photo() -> str:
    path = ASSETS / "bahan-dapur.webp"
    return b64encode(path.read_bytes()).decode("ascii") if path.is_file() else ""


@st.cache_data(show_spinner=False)
def brand_logo() -> str:
    if not LOGO.is_file():
        return '<p class="resnap-brand">ReSnap</p>'

    buffer = BytesIO()
    with Image.open(LOGO) as image:
        image.thumbnail((224, 224), Image.Resampling.LANCZOS)
        image.save(buffer, format="WEBP")
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="resnap-logo" src="data:image/webp;base64,{encoded}" alt="ReSnap">'


def show_hero():
    st.html(ASSETS / "resnap.css")
    photo = hero_photo()
    style = f"style=\"--resnap-photo: url('data:image/webp;base64,{photo}')\"" if photo else ""
    st.html(
        f"""
        <header class="resnap-hero" {style}>
            <div class="resnap-hero-copy">
                {brand_logo()}
                <h1>Recipe <span>From a Snap.</span></h1>
                <p class="resnap-intro">Pindai bahan yang ada di dapur dan temukan resep yang bisa kamu masak.</p>
            </div>
        </header>
        """
    )


def show_photo_input():
    with st.container(border=True, key="photo_panel"):
        st.subheader("Foto bahan")
        source = st.segmented_control(
            "Sumber foto", ("Unggah foto", "Kamera"), default="Unggah foto",
            selection_mode="single", label_visibility="collapsed",
        )
        if source == "Kamera":
            st.caption(
                "Kamera di browser HP memerlukan HTTPS. Jika izin kamera tidak muncul saat "
                "membuka alamat HTTP lokal, ambil foto dengan kamera HP lalu pilih Unggah foto."
            )
            upload = st.camera_input("Ambil foto bahan")
        else:
            upload = st.file_uploader("Pilih foto bahan", type=("jpg", "jpeg", "png"))
        photo = read_photo(upload)
        if photo is not None:
            if st.button("Pindai bahan", type="primary", width="stretch"):
                scan_photo(photo)
            show_detection(photo)


def main():
    show_hero()
    with st.container(key="workspace"):
        photo_column, recipe_column = st.columns((1, 1.12), gap="medium")
        with photo_column:
            show_photo_input()
        with recipe_column:
            with st.container(border=True, key="ingredient_panel"):
                st.subheader("Bahan yang tersedia")
                selected = st.multiselect(
                    "Tambah atau hapus bahan sesuai isi dapurmu",
                    options=sorted(INGREDIENTS, key=display_name),
                    format_func=display_name,
                    placeholder="Cari bahan, misalnya telur",
                    key="ingredients",
                )
            with st.container(key="recipe_section"):
                st.subheader("Rekomendasi resep")
                show_recipes(selected)


if __name__ == "__main__":
    main()
