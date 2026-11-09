# Automatic Number Plate Recognition

Fork ini mengembangkan proyek ANPR berbasis YOLOv7 Tiny dan EasyOCR dengan alur penggunaan **unggah foto kendaraan**, bukan webcam.

## Cara kerja

1. Model YOLOv7 Tiny mendeteksi lokasi plat nomor.
2. Area plat dipotong dan diproses dengan Real-Time Super Resolution.
3. EasyOCR membaca karakter pada plat.
4. Aplikasi menampilkan gambar hasil deteksi dan teks plat yang terbaca.

## Model dan training

Model deteksi yang digunakan aplikasi berada di `model/best.onnx`. Notebook training dan data preparation asli tetap tersedia di folder `notebook/`.

Jika ingin melatih model dengan dataset sendiri, ikuti notebook:
- `notebook/(1) Data Preparation_ Plate Detection.ipynb`
- `notebook/(2) Training YOLOv7 Tiny_ Plate Detection.ipynb`

Setelah training, export model hasil training ke ONNX dan simpan/ganti file `model/best.onnx`. Pastikan model memiliki format input dan output yang sesuai dengan pipeline pada `plate_recognition.py`.

EasyOCR pada versi ini masih menggunakan model pretrained bahasa Inggris (alfanumerik). Training ulang OCR merupakan pekerjaan terpisah dan tidak otomatis dilakukan ketika menjalankan aplikasi.

## Instalasi

Disarankan menggunakan Python 3.8 sesuai lingkungan proyek asal.

```bash
git clone https://github.com/jackyjeky/plate-detection.git
cd plate-detection
python -m venv venv
```

Aktifkan virtual environment:

Windows:
```bash
venv\\Scripts\\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

Install dependensi:

```bash
pip install -r requirements.txt
```

## Menjalankan aplikasi

```bash
streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit, unggah foto JPG/PNG, lalu tekan **Deteksi plat nomor**.

Secara default aplikasi menggunakan CPU. Aktifkan opsi CUDA di sidebar hanya jika ONNX Runtime GPU dan TensorFlow GPU sudah dikonfigurasi dengan benar.

## Menjalankan inferensi melalui CLI (opsional)

Mode CLI asli tetap tersedia:

```bash
python main.py --source "./img/Cars1.png" --threshold 0.55
```

## Catatan dan batasan

- Hasil OCR bergantung pada resolusi, sudut, pencahayaan, dan kualitas foto.
- Model deteksi yang disertakan adalah model pretrained dari repositori asal, bukan model yang otomatis dilatih ulang.
- Dataset dan hasil training perlu dievaluasi sebelum digunakan untuk kebutuhan nyata.
- Sistem saat ini mengembalikan satu teks plat dari pemrosesan gambar; dukungan multi-plat dalam satu foto dapat ditingkatkan pada tahap berikutnya.

## Kredit

Berdasarkan proyek asli [daffatm/plate-detection](https://github.com/daffatm/plate-detection). Model deteksi menggunakan [YOLOv7](https://github.com/WongKinYiu/yolov7), OCR menggunakan [EasyOCR](https://github.com/JaidedAI/EasyOCR), dan super-resolution menggunakan [Real-Time Super-Resolution](https://github.com/braindotai/Real-Time-Super-Resolution).
