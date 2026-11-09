from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from plate_recognition import PlateRecognition
from super_resolution import SuperResolution


MODEL_PATH = Path("model/best.onnx")


@st.cache_resource(show_spinner=False)
def load_anpr_model(model_path: str, use_cuda: bool):
    enhancer = SuperResolution()
    recognizer = PlateRecognition(model_path, enhancer, cuda=use_cuda)
    return recognizer


st.set_page_config(
    page_title="Automatic Number Plate Recognition",
    page_icon="🚘",
    layout="centered",
)

st.title("Automatic Number Plate Recognition")
st.write(
    "Unggah foto kendaraan untuk mendeteksi lokasi plat nomor "
    "dan membaca karakter yang tertera."
)

with st.sidebar:
    st.header("Pengaturan")
    confidence_threshold = st.slider(
        "Detection threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.55,
        step=0.05,
        help="Semakin tinggi nilainya, semakin ketat model menerima deteksi.",
    )
    use_cuda = st.checkbox(
        "Gunakan CUDA / GPU",
        value=False,
        help="Aktifkan hanya jika ONNX Runtime dan TensorFlow GPU sudah dikonfigurasi.",
    )

uploaded_file = st.file_uploader(
    "Pilih foto kendaraan",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if not MODEL_PATH.exists():
    st.error(
        f"Model tidak ditemukan: {MODEL_PATH}. "
        "Pastikan file hasil training/export tersedia di folder model."
    )
    st.stop()

if uploaded_file is None:
    st.info("Unggah foto berformat JPG atau PNG untuk memulai deteksi.")
    st.stop()

file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if input_image is None:
    st.error("File tidak dapat dibaca sebagai gambar. Coba gunakan file JPG atau PNG lain.")
    st.stop()

st.subheader("Foto yang diunggah")
st.image(
    cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
    caption=uploaded_file.name,
    use_container_width=True,
)

if st.button("Deteksi plat nomor", type="primary", use_container_width=True):
    try:
        with st.spinner("Memuat model dan memproses foto..."):
            recognizer = load_anpr_model(str(MODEL_PATH), use_cuda)
            result_image, plate_number = recognizer.anpr(
                input_image.copy(),
                confidence_threshold,
            )

        st.subheader("Hasil deteksi")
        st.image(
            cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
            caption="Foto dengan bounding box hasil deteksi",
            use_container_width=True,
        )

        if plate_number and plate_number != "not detected":
            st.success(f"Plat nomor terbaca: **{plate_number}**")
        elif plate_number == "not detected":
            st.warning(
                "Area plat terdeteksi, tetapi karakter plat belum berhasil dibaca OCR."
            )
        else:
            st.warning(
                "Plat nomor tidak ditemukan. Coba foto yang lebih jelas "
                "atau sesuaikan detection threshold."
            )

    except Exception as error:
        st.error(f"Proses deteksi gagal: {error}")
        st.caption(
            "Periksa dependensi, file model, dan konfigurasi CUDA. "
            "Untuk pengujian awal, gunakan mode CPU."
        )
