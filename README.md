# Smart Recipe

Smart Recipe memindai bahan masakan dari foto dengan YOLO dan mencari resep di dataset lokal. Kamu bisa memperbaiki daftar bahan sebelum memilih resep.

## Jalankan di komputer

Dari direktori proyek, gunakan Python 3.11 atau 3.12 dan jalankan perintah berikut di macOS atau Linux.

Di macOS dengan Homebrew, jika `python3.12 --version` menghasilkan `command not found`, pasang Python 3.12 dengan `brew install python@3.12` terlebih dahulu.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Jika memakai Python 3.11, ganti `python3.12` dengan `python3.11`. Buka [http://localhost:8501](http://localhost:8501) atau alamat yang dicetak Streamlit. Pastikan `models/best.pt` dan `data/recipes.json.gz` berada di lokasi tersebut. Internet diperlukan untuk memasang dependensi, sementara pemindaian dan rekomendasi berjalan lokal.

## Gunakan aplikasi

1. Unggah foto JPG atau PNG, atau ambil foto dari opsi **Kamera**, lalu pilih **Pindai bahan**. Batas foto adalah 10 MB dan 24 megapiksel.
2. Tinjau kotak deteksi dan koreksi daftar bahan. Kamu juga bisa memilih bahan langsung tanpa foto.
3. Buka kartu resep untuk melihat kecocokan bahan utama, bahan yang perlu disiapkan, dan langkah memasak.

Untuk akses dari HP atau deployment, jalankan Streamlit di host yang bisa dijangkau perangkat dan sertakan bobot serta dataset. Gunakan HTTPS agar browser HP dapat meminta izin kamera. Unggah foto tetap dapat dipakai jika akses kamera tidak tersedia.

## Cek pipeline

```sh
python resnap_pipeline.py contoh/foto_udang.jpg
```

`scan()` menerima path foto, bytes, atau array BGR OpenCV. Hasilnya memuat `deteksi`, `bahan`, dan `resep`. Untuk menghitung rekomendasi setelah daftar bahan diedit, panggil `rekomendasi()` dengan ID kelas seperti `egg` atau `tomato`.
