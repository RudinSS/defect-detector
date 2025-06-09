import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
import tempfile
from ultralytics import YOLO
import requests
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Defect Pakaian",
    page_icon="👔",
    layout="wide"
)

# Judul aplikasi
st.title('👔 Deteksi Defect Pakaian dengan YOLOv11')
st.write('Aplikasi ini mendeteksi defect pada pakaian menggunakan model YOLOv11')

# Fungsi untuk download model dari URL (jika tidak ada lokal)
@st.cache_data
def download_model():
    model_url = "https://your-model-storage-url.com/best.pt"  # Ganti dengan URL model Anda
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... Ini mungkin memakan waktu beberapa menit."):
            try:
                response = requests.get(model_url)
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                st.success("Model berhasil didownload!")
            except Exception as e:
                st.error(f"Gagal download model: {e}")
                return None
    return model_path

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    # Coba beberapa lokasi model
    possible_paths = [
        "runs/detect/train52/weights/best.pt",  # Path lokal development
        "best.pt",  # Path di root untuk deployment
        "model/best.pt"  # Path alternatif
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    # Jika tidak ada model lokal, coba download
    if model_path is None:
        st.warning("Model tidak ditemukan secara lokal. Mencoba download...")
        model_path = download_model()
    
    if model_path is None:
        st.error("Model tidak dapat dimuat. Pastikan file model tersedia.")
        return None
        
    try:
        model = YOLO(model_path)
        st.success(f"Model berhasil dimuat dari: {model_path}")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Fungsi deteksi
def detect_defects(model, image, conf_threshold):
    try:
        results = model(image, conf=conf_threshold)
        annotated_image = results[0].plot()

        boxes = results[0].boxes
        names = results[0].names

        detections = []
        if boxes is not None and len(boxes) > 0 and boxes.xyxy is not None:
            for i in range(len(boxes)):
                box = boxes.xyxy[i].tolist()
                conf = boxes.conf[i].item()
                cls_id = int(boxes.cls[i].item())
                cls_name = names[cls_id] if names and cls_id in names else str(cls_id)
                detections.append({
                    "name": cls_name,
                    "confidence": conf,
                    "xmin": box[0],
                    "ymin": box[1],
                    "xmax": box[2],
                    "ymax": box[3],
                })

        return annotated_image, detections
    except Exception as e:
        st.error(f"Error dalam deteksi: {e}")
        return image, []

# Muat model
model = load_model()

# Sidebar pengaturan
st.sidebar.title("⚙️ Pengaturan")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
input_source = st.sidebar.radio("Pilih Sumber Input", ["Upload Gambar", "Upload Video", "Kamera"])

# Informasi deployment
if st.sidebar.button("ℹ️ Info Aplikasi"):
    st.sidebar.info("""
    **Aplikasi Deteksi Defect Pakaian**
    
    - Model: YOLOv11
    - Deploy Version: 1.0
    - Akses via HP: ✅ Supported
    - Kamera HP: ⚠️ Terbatas (tergantung browser)
    """)

# Deteksi via Gambar (prioritas utama untuk mobile)
if input_source == "Upload Gambar":
    st.header("📸 Deteksi via Upload Gambar")
    
    # Info untuk pengguna mobile
    st.info("💡 Tip: Untuk hasil terbaik di HP, gunakan gambar dengan resolusi 640x640 atau lebih kecil")
    
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Simpan gambar ke session state
        if 'uploaded_image' not in st.session_state or st.session_state['uploaded_image_name'] != uploaded_file.name:
            image = Image.open(uploaded_file)
            # Resize jika terlalu besar (untuk performa mobile)
            if image.width > 1024 or image.height > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            st.session_state['uploaded_image'] = np.array(image)
            st.session_state['uploaded_image_name'] = uploaded_file.name
            
        img_array = st.session_state['uploaded_image']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gambar Asli")
            st.image(img_array, channels="RGB", use_column_width=True)

        if model:
            with st.spinner('🔍 Mendeteksi defect...'):
                img_with_boxes, detections = detect_defects(model, img_array, confidence_threshold)
                
                with col2:
                    st.subheader("Hasil Deteksi")
                    st.image(img_with_boxes, channels="RGB", use_column_width=True)

                if detections:
                    st.subheader(f"🎯 Deteksi Defect (Confidence ≥ {confidence_threshold})")
                    
                    # Tampilan yang mobile-friendly
                    for i, det in enumerate(detections, 1):
                        st.write(f"**{i}. {det['name']}** - Confidence: {det['confidence']:.2f}")
                    
                    # Tabel detail (collapsed by default di mobile)
                    with st.expander("📊 Detail Deteksi"):
                        st.dataframe(detections, use_container_width=True)
                else:
                    st.info("❌ Tidak ada defect yang terdeteksi.")

# Deteksi via Video
elif input_source == "Upload Video":
    st.header("🎥 Deteksi via Upload Video")
    st.warning("⚠️ Fitur video mungkin lambat di perangkat mobile dengan spesifikasi rendah")
    
    uploaded_video = st.file_uploader("Pilih file video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Limit ukuran video untuk mobile
        if uploaded_video.size > 50 * 1024 * 1024:  # 50MB
            st.error("❌ File video terlalu besar (max 50MB untuk deployment)")
        else:
            # Simpan video ke file sementara
            if 'video_path' not in st.session_state or st.session_state['video_name'] != uploaded_video.name:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                st.session_state['video_path'] = tfile.name
                st.session_state['video_name'] = uploaded_video.name
                
                # Baca informasi video
                cap = cv2.VideoCapture(st.session_state['video_path'])
                st.session_state['video_fps'] = int(cap.get(cv2.CAP_PROP_FPS))
                st.session_state['video_frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
            # Gunakan video yang sudah disimpan
            cap = cv2.VideoCapture(st.session_state['video_path'])

            if not cap.isOpened():
                st.error("❌ Gagal membuka video.")
            else:
                fps = st.session_state['video_fps']
                frame_count = st.session_state['video_frame_count']

                st.write(f"📹 FPS: {fps}, Total Frame: {frame_count}")
                frame_number = st.slider('Frame', 0, frame_count-1, 0)
                
                # Cache frame untuk performa
                if 'current_frame_number' not in st.session_state:
                    st.session_state['current_frame_number'] = frame_number
                    
                if frame_number != st.session_state['current_frame_number'] or 'current_frame' not in st.session_state:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    if ret:
                        st.session_state['current_frame'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.session_state['current_frame_number'] = frame_number
                
                if 'current_frame' in st.session_state:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(st.session_state['current_frame'], caption=f'Frame {frame_number}', use_column_width=True)
                    
                    if model:
                        with st.spinner('🔍 Mendeteksi defect...'):
                            frame_with_boxes, detections = detect_defects(model, st.session_state['current_frame'], confidence_threshold)
                            
                            with col2:
                                st.subheader("Hasil Deteksi")
                                st.image(frame_with_boxes, use_column_width=True)

                            if detections:
                                st.subheader(f"🎯 Deteksi Defect (Confidence ≥ {confidence_threshold})")
                                for i, det in enumerate(detections, 1):
                                    st.write(f"**{i}. {det['name']}** - Confidence: {det['confidence']:.2f}")
                            else:
                                st.info("❌ Tidak ada defect yang terdeteksi.")

            cap.release()

# Deteksi via Kamera (dengan warning untuk mobile)
elif input_source == "Kamera":
    st.header("📱 Deteksi via Kamera")
    
    st.warning("""
    ⚠️ **Catatan Penting untuk Pengguna Mobile:**
    - Fitur kamera mungkin tidak berfungsi di semua browser mobile
    - Disarankan menggunakan Chrome atau Safari terbaru
    - Pastikan memberikan izin akses kamera saat diminta
    - Untuk hasil terbaik, gunakan fitur "Upload Gambar" di HP
    """)
    
    if st.button("🔄 Coba Akses Kamera"):
        st.info("Klik tombol 'Mulai Kamera' jika tersedia, atau gunakan fitur upload gambar sebagai alternatif.")
    
    # Sisanya sama seperti kode kamera sebelumnya...
    # (Kode kamera yang sudah diperbaiki sebelumnya)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**📱 Mobile Friendly**")
st.sidebar.markdown("✅ Upload Gambar")
st.sidebar.markdown("⚠️ Upload Video (Terbatas)")
st.sidebar.markdown("❓ Kamera (Tergantung Browser)")
st.sidebar.markdown("---")
st.sidebar.write("© 2025 Deteksi Defect Pakaian")

# Cleanup function
def cleanup():
    if 'video_path' in st.session_state and os.path.exists(st.session_state['video_path']):
        os.unlink(st.session_state['video_path'])

if 'cleanup' not in st.session_state:
    import atexit
    atexit.register(cleanup)
    st.session_state['cleanup'] = True