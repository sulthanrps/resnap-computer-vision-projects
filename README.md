# ReSnap - Recipe from a Snap !

Punya banyak bahan sisa tapi bingung mau dimasak apa ? Unggah foto bahan yang ada, lalu Smart Recipe akan mencari resep yang bisa kamu buat. Say no more to wasting food !.

Aplikasi ini memakai Streamlit, model YOLOv8s dengan 30 kelas bahan, dan dataset lokal berisi 14.658 resep Cookpad Indonesia.

## Jalankan aplikasi

Buka terminal di folder proyek.

### macOS dan Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Windows (PowerShell)

Jalankan perintah ini dari folder proyek. Python di dalam `.venv` dipanggil langsung.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Kalau perintah `py` tidak tersedia tetapi `python --version` berhasil, pakai `python -m venv .venv` untuk langkah pertama.

### Coba foto contoh

Buka [localhost:8501](http://localhost:8501). Untuk percobaan pertama, unggah [foto udang contoh](contoh/foto_udang.jpg) dan klik **Pindai bahan**. Aplikasi menampilkan hasil berupa kotak deteksi pada foto, daftar bahan yang bisa diedit, dan rekomendasi resep di bawahnya.

Skor pada kartu resep menunjukkan berapa banyak bahan utama yang sudah anda miliki dan terdeteksi oleh sistem. Buka kartunya untuk melihat daftar bahan lengkap dan langkah memasak. Kalau fotonya sulit dikenali, pilih bahan secara manual. Aplikasi menerima JPG dan PNG dengan batas 10 MB dan 24 megapiksel.

## Cek deteksi dari terminal

Di macOS atau Linux, jalankan ini setelah mengaktifkan `.venv`.

```sh
python resnap_pipeline.py contoh/foto_udang.jpg --gambar /tmp/resnap-hasil.jpg
```

Di Windows, gunakan PowerShell.

```powershell
.\.venv\Scripts\python.exe resnap_pipeline.py contoh\foto_udang.jpg --gambar "$env:TEMP\resnap-hasil.jpg"
```

Kedua perintah mencetak hasil deteksi dan resep sebagai JSON, lalu menyimpan foto dengan kotak deteksi di folder sementara. Ganti path foto untuk mencoba gambar sendiri.

Model dibaca dari `models/best.pt` dan resep dari `data/recipes.json.gz`. Setelah dependensi terpasang, pemindaian dan pencarian resep berjalan lokal.
