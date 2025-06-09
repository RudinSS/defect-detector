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
import threading
import queue

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

# Class untuk menghandle kamera
class CameraHandler:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.camera_thread = None
        
    def start_camera(self, camera_index=0):
        """Mulai kamera"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                return False
            
            # Set resolusi kamera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.camera_thread = threading.Thread(target=self._capture_frames)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            return True
        except Exception as e:
            st.error(f"Gagal memulai kamera: {e}")
            return False
    
    def _capture_frames(self):
        """Capture frames di thread terpisah"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Kosongkan queue jika penuh
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Tambahkan frame baru
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            else:
                break
    
    def get_frame(self):
        """Ambil frame terbaru"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop_camera(self):
        """Hentikan kamera"""
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1)
        if self.cap:
            self.cap.release()

# Inisialisasi camera handler
if 'camera_handler' not in st.session_state:
    st.session_state.camera_handler = CameraHandler()

# Muat model
model = load_model()

# Sidebar pengaturan
st.sidebar.title("⚙️ Pengaturan")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Pilihan sumber input
input_source = st.sidebar.radio(
    "Pilih Sumber Input", 
    ["Kamera Real-time", "Upload Gambar", "Upload Video"]
)

# Tambahkan informasi deployment
with st.sidebar.expander("ℹ️ Info Deployment"):
    st.write("""
    **Untuk deployment di Streamlit Cloud:**
    1. Upload model ke Google Drive
    2. Set MODEL_URL di secrets
    3. Kamera mungkin tidak tersedia di cloud deployment
    """)

# Kamera Real-time
if input_source == "Kamera Real-time":
    st.header("📹 Deteksi Real-time dengan Kamera")
    
    # Kontrol kamera
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🎥 Mulai Kamera", type="primary"):
            if st.session_state.camera_handler.start_camera():
                st.session_state.camera_started = True
                st.success("Kamera dimulai!")
            else:
                st.error("Gagal memulai kamera. Pastikan kamera tersedia dan tidak digunakan aplikasi lain.")
    
    with col2:
        if st.button("⏹️ Hentikan Kamera"):
            st.session_state.camera_handler.stop_camera()
            st.session_state.camera_started = False
            st.info("Kamera dihentikan.")
    
    with col3:
        camera_index = st.selectbox("Pilih Kamera", [0, 1, 2], help="0 = Kamera default, 1 = Kamera eksternal")
    
    # Tambahkan pengaturan deteksi
    col1, col2 = st.columns(2)
    with col1:
        detection_frequency = st.slider("Frekuensi Deteksi (detik)", 0.1, 2.0, 0.5, 0.1)
    with col2:
        show_fps = st.checkbox("Tampilkan FPS", True)
    
    # Area untuk menampilkan video
    if 'camera_started' in st.session_state and st.session_state.camera_started:
        # Placeholder untuk video dan hasil
        video_placeholder = st.empty()
        detection_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Inisialisasi variabel untuk FPS
        fps_counter = 0
        start_time = time.time()
        last_detection_time = 0
        
        # Loop untuk menampilkan video real-time
        while st.session_state.get('camera_started', False):
            frame = st.session_state.camera_handler.get_frame()
            
            if frame is not None:
                # Convert BGR ke RGB untuk Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Hitung FPS
                fps_counter += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if elapsed_time >= 1.0:
                    fps = fps_counter / elapsed_time
                    fps_counter = 0
                    start_time = current_time
                
                # Lakukan deteksi berdasarkan frekuensi yang ditentukan
                if current_time - last_detection_time >= detection_frequency:
                    if model:
                        try:
                            frame_with_boxes, detections = detect_defects(model, frame_rgb, confidence_threshold)
                            last_detection_time = current_time
                            
                            # Tampilkan hasil deteksi
                            with detection_placeholder.container():
                                if detections:
                                    st.subheader(f"🔍 Defect Terdeteksi: {len(detections)} item")
                                    
                                    # Buat dataframe untuk hasil deteksi
                                    detection_df = []
                                    for det in detections:
                                        detection_df.append({
                                            "Jenis": det['name'],
                                            "Confidence": f"{det['confidence']:.3f}",
                                            "Koordinat": f"({det['xmin']:.0f}, {det['ymin']:.0f}) - ({det['xmax']:.0f}, {det['ymax']:.0f})"
                                        })
                                    
                                    st.dataframe(detection_df, use_container_width=True)
                                else:
                                    st.info("✅ Tidak ada defect terdeteksi")
                            
                            # Tampilkan frame dengan deteksi
                            display_frame = frame_with_boxes
                            
                        except Exception as e:
                            st.error(f"Error dalam deteksi: {e}")
                            display_frame = frame_rgb
                    else:
                        display_frame = frame_rgb
                        with detection_placeholder.container():
                            st.warning("Model tidak tersedia - hanya menampilkan kamera")
                else:
                    # Gunakan frame tanpa deteksi untuk menghemat komputasi
                    display_frame = frame_rgb
                
                # Tambahkan informasi FPS ke frame jika diaktifkan
                if show_fps and 'fps' in locals():
                    display_frame_copy = display_frame.copy()
                    # Tambahkan teks FPS (simulasi, karena cv2.putText tidak tersedia di Streamlit)
                    # Kita akan menampilkan FPS di bagian terpisah
                
                # Tampilkan frame
                with video_placeholder.container():
                    st.image(display_frame, channels="RGB", use_container_width=True)
                
                # Tampilkan statistik
                if show_fps and 'fps' in locals():
                    with stats_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("FPS", f"{fps:.1f}")
                        with col2:
                            st.metric("Resolusi", f"{frame.shape[1]}x{frame.shape[0]}")
                        with col3:
                            st.metric("Status", "🟢 Aktif")
                
                # Delay kecil untuk mengurangi beban CPU
                time.sleep(0.03)  # ~30 FPS
            else:
                time.sleep(0.1)  # Tunggu frame berikutnya
    else:
        st.info("👆 Klik 'Mulai Kamera' untuk memulai deteksi real-time")
        st.write("""
        **Catatan:**
        - Pastikan kamera tidak digunakan oleh aplikasi lain
        - Berikan izin akses kamera pada browser
        - Gunakan browser modern (Chrome, Firefox, Edge)
        - Untuk performa terbaik, gunakan kamera dengan resolusi 640x480
        """)

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

# Fungsi cleanup untuk kamera
def cleanup_camera():
    if 'camera_handler' in st.session_state:
        st.session_state.camera_handler.stop_camera()

# Cleanup otomatis saat aplikasi ditutup
if 'cleanup_registered' not in st.session_state:
    import atexit
    atexit.register(cleanup_camera)
    st.session_state['cleanup_registered'] = True

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Petunjuk Penggunaan")
st.sidebar.markdown("""
**Kamera Real-time:**
1. Klik 'Mulai Kamera'
2. Berikan izin akses kamera
3. Atur confidence threshold
4. Lihat hasil deteksi real-time

**Dependencies:**
```
pip install streamlit opencv-python ultralytics pillow gdown
```
""")

st.sidebar.markdown("---")
st.sidebar.write("© 2025 Deteksi Defect Pakaian")
st.sidebar.write("📹 Dengan Kamera Real-time!")
