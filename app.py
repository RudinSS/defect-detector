import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
import tempfile
import requests
from io import BytesIO
import gdown

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Defect Pakaian",
    page_icon="👕",
    layout="wide"
)

# Judul aplikasi
st.title('👕 Deteksi Defect Pakaian dengan YOLOv11')
st.write('Aplikasi ini mendeteksi defect pada pakaian menggunakan model YOLOv11')

# Fungsi untuk download model dari Google Drive atau URL
@st.cache_resource
def download_model():
    """Download model dari Google Drive atau URL eksternal"""
    model_url = os.getenv('MODEL_URL')  # Set di environment variables
    model_path = "best.pt"
    
    if model_url:
        try:
            with st.spinner("Mengunduh model..."):
                if 'drive.google.com' in model_url:
                    # Ekstrak file ID dari Google Drive URL
                    file_id = model_url.split('/d/')[1].split('/')[0]
                    gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)
                else:
                    # Download dari URL biasa
                    response = requests.get(model_url)
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                st.success("Model berhasil diunduh!")
                return model_path
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            return None
    else:
        st.warning("URL model tidak ditemukan. Set MODEL_URL di environment variables.")
        return None

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    try:
        # Coba load model lokal terlebih dahulu
        model_path = "best.pt"
        if not os.path.exists(model_path):
            # Jika tidak ada, coba download
            model_path = download_model()
            if not model_path:
                return None
        
        # Import YOLO di sini untuk menghindari error jika ultralytics tidak tersedia
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            st.success("Model berhasil dimuat!")
            return model
        except ImportError:
            st.error("Ultralytics tidak terinstall. Jalankan: pip install ultralytics")
            return None
            
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.info("Pastikan file model tersedia atau set MODEL_URL di environment variables")
        return None

# Fungsi deteksi (dengan fallback jika model tidak tersedia)
def detect_defects(model, image, conf_threshold):
    if model is None:
        # Return dummy detection untuk demo
        return image, []
    
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
                    "confidence": round(conf, 3),
                    "xmin": round(box[0], 1),
                    "ymin": round(box[1], 1),
                    "xmax": round(box[2], 1),
                    "ymax": round(box[3], 1),
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

# Untuk deployment, hanya gunakan upload gambar dan video
input_source = st.sidebar.radio(
    "Pilih Sumber Input", 
    ["Upload Gambar", "Upload Video", "Demo Gambar"]
)

# Tambahkan informasi deployment
with st.sidebar.expander("ℹ️ Info Deployment"):
    st.write("""
    **Untuk deployment di Streamlit Cloud:**
    1. Upload model ke Google Drive
    2. Set MODEL_URL di secrets
    3. Kamera tidak tersedia di cloud deployment
    """)

# Demo gambar untuk testing
if input_source == "Demo Gambar":
    st.header("🖼️ Demo dengan Gambar Contoh")
    
    # URL gambar demo (gunakan gambar dari internet)
    demo_urls = {
        "Contoh 1": "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=500",
        "Contoh 2": "https://images.unsplash.com/photo-1489987707025-afc232f7ea0f?w=500",
        "Contoh 3": "https://images.unsplash.com/photo-1562157873-818bc0726f68?w=500"
    }
    
    selected_demo = st.selectbox("Pilih gambar demo:", list(demo_urls.keys()))
    
    try:
        with st.spinner("Memuat gambar demo..."):
            response = requests.get(demo_urls[selected_demo])
            image = Image.open(BytesIO(response.content))
            img_array = np.array(image)
            
        if model:
            with st.spinner('Mendeteksi defect...'):
                img_with_boxes, detections = detect_defects(model, img_array, confidence_threshold)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Gambar Asli")
                    st.image(img_array, use_container_width=True)
                
                with col2:
                    st.subheader("Hasil Deteksi")
                    st.image(img_with_boxes, use_container_width=True)
                
                if detections:
                    st.subheader(f"Deteksi Defect (Confidence ≥ {confidence_threshold})")
                    st.dataframe(detections, use_container_width=True)
                else:
                    st.info("Tidak ada defect yang terdeteksi.")
        else:
            st.image(img_array, caption="Gambar Demo (Model tidak tersedia)", use_container_width=True)
            
    except Exception as e:
        st.error(f"Gagal memuat gambar demo: {e}")

# Deteksi via Upload Gambar
elif input_source == "Upload Gambar":
    st.header("📁 Deteksi via Upload Gambar")
    
    uploaded_file = st.file_uploader(
        "Pilih file gambar...", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload gambar pakaian untuk mendeteksi defect"
    )

    if uploaded_file is not None:
        # Validasi ukuran file (maksimal 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("Ukuran file terlalu besar! Maksimal 10MB.")
        else:
            try:
                # Simpan gambar ke session state
                if 'uploaded_image' not in st.session_state or st.session_state['uploaded_image_name'] != uploaded_file.name:
                    image = Image.open(uploaded_file)
                    # Resize jika terlalu besar
                    if image.size[0] > 1920 or image.size[1] > 1920:
                        image.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
                    st.session_state['uploaded_image'] = np.array(image)
                    st.session_state['uploaded_image_name'] = uploaded_file.name
                    
                img_array = st.session_state['uploaded_image']
                
                if model:
                    with st.spinner('Mendeteksi defect...'):
                        img_with_boxes, detections = detect_defects(model, img_array, confidence_threshold)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Gambar Asli")
                            st.image(img_array, use_container_width=True)
                        
                        with col2:
                            st.subheader("Hasil Deteksi")
                            st.image(img_with_boxes, use_container_width=True)
                        
                        if detections:
                            st.subheader(f"Deteksi Defect (Confidence ≥ {confidence_threshold})")
                            st.dataframe(detections, use_container_width=True)
                            
                            # Statistik deteksi
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Defect", len(detections))
                            with col2:
                                avg_conf = np.mean([d['confidence'] for d in detections])
                                st.metric("Avg Confidence", f"{avg_conf:.3f}")
                            with col3:
                                unique_classes = len(set(d['name'] for d in detections))
                                st.metric("Jenis Defect", unique_classes)
                        else:
                            st.success("✅ Tidak ada defect yang terdeteksi.")
                else:
                    st.image(img_array, caption="Gambar yang diupload (Model tidak tersedia)", use_container_width=True)
                    st.warning("Model tidak tersedia. Hasil deteksi tidak dapat ditampilkan.")
                    
            except Exception as e:
                st.error(f"Error memproses gambar: {e}")

# Deteksi via Upload Video
elif input_source == "Upload Video":
    st.header("🎥 Deteksi via Upload Video")
    
    uploaded_video = st.file_uploader(
        "Pilih file video...", 
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload video pakaian untuk mendeteksi defect"
    )

    if uploaded_video is not None:
        # Validasi ukuran file (maksimal 50MB)
        if uploaded_video.size > 50 * 1024 * 1024:
            st.error("Ukuran file terlalu besar! Maksimal 50MB.")
        else:
            try:
                # Simpan video ke file sementara
                if 'video_path' not in st.session_state or st.session_state['video_name'] != uploaded_video.name:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    tfile.close()
                    st.session_state['video_path'] = tfile.name
                    st.session_state['video_name'] = uploaded_video.name
                    
                    # Baca informasi video
                    cap = cv2.VideoCapture(st.session_state['video_path'])
                    st.session_state['video_fps'] = int(cap.get(cv2.CAP_PROP_FPS))
                    st.session_state['video_frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                cap = cv2.VideoCapture(st.session_state['video_path'])

                if not cap.isOpened():
                    st.error("Gagal membuka video.")
                else:
                    fps = st.session_state['video_fps']
                    frame_count = st.session_state['video_frame_count']
                    duration = frame_count / fps if fps > 0 else 0

                    # Info video
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("FPS", fps)
                    with col2:
                        st.metric("Total Frame", frame_count)
                    with col3:
                        st.metric("Durasi", f"{duration:.1f}s")
                    
                    frame_number = st.slider('Pilih Frame', 0, frame_count-1, 0, help="Geser untuk memilih frame yang ingin dianalisis")
                    
                    # Ambil frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        if model:
                            with st.spinner('Mendeteksi defect...'):
                                frame_with_boxes, detections = detect_defects(model, frame_rgb, confidence_threshold)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader(f"Frame Asli ({frame_number}/{frame_count-1})")
                                    st.image(frame_rgb, use_container_width=True)
                                
                                with col2:
                                    st.subheader("Hasil Deteksi")
                                    st.image(frame_with_boxes, use_container_width=True)
                                
                                if detections:
                                    st.subheader(f"Deteksi Defect (Confidence ≥ {confidence_threshold})")
                                    st.dataframe(detections, use_container_width=True)
                                else:
                                    st.info("Tidak ada defect yang terdeteksi pada frame ini.")
                        else:
                            st.image(frame_rgb, caption=f"Frame {frame_number} (Model tidak tersedia)", use_container_width=True)
                    else:
                        st.error("Gagal membaca frame video.")

                cap.release()
                
            except Exception as e:
                st.error(f"Error memproses video: {e}")

# Bersihkan file sementara
def cleanup():
    if 'video_path' in st.session_state and os.path.exists(st.session_state['video_path']):
        try:
            os.unlink(st.session_state['video_path'])
        except:
            pass

# Cleanup otomatis
if 'cleanup_registered' not in st.session_state:
    import atexit
    atexit.register(cleanup)
    st.session_state['cleanup_registered'] = True

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Petunjuk Deployment")
st.sidebar.markdown("""
1. **Upload model ke Google Drive**
2. **Set MODEL_URL di Streamlit secrets**
3. **Install dependencies:**
   ```
   pip install streamlit opencv-python ultralytics pillow gdown
   ```
""")

st.sidebar.markdown("---")
st.sidebar.write("© 2025 Deteksi Defect Pakaian")
st.sidebar.write("🚀 Ready for Streamlit Cloud!")
