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
import asyncio
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
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
st.write('Aplikasi ini mendeteksi defect pada pakaian menggunakan model YOLOv11 dengan WebRTC')

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

# Class VideoTransformer untuk WebRTC
class DefectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.conf_threshold = 0.5
        self.detection_frequency = 0.5  # deteksi setiap 0.5 detik
        self.last_detection_time = 0
        self.last_detections = []
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Queue untuk sharing hasil deteksi dengan main thread
        self.detection_queue = queue.Queue(maxsize=1)
        
    def set_model(self, model):
        self.model = model
        
    def set_conf_threshold(self, threshold):
        self.conf_threshold = threshold
        
    def set_detection_frequency(self, frequency):
        self.detection_frequency = frequency
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Hitung FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = current_time
        
        # Convert BGR ke RGB untuk model
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Lakukan deteksi berdasarkan frekuensi
        if current_time - self.last_detection_time >= self.detection_frequency:
            if self.model is not None:
                try:
                    _, detections = detect_defects(self.model, img_rgb, self.conf_threshold)
                    self.last_detections = detections
                    self.last_detection_time = current_time
                    
                    # Simpan hasil deteksi ke queue untuk main thread
                    try:
                        if not self.detection_queue.empty():
                            self.detection_queue.get_nowait()  # Hapus hasil lama
                        self.detection_queue.put_nowait({
                            'detections': detections,
                            'fps': self.current_fps,
                            'timestamp': current_time
                        })
                    except queue.Full:
                        pass
                        
                except Exception as e:
                    print(f"Error in detection: {e}")
        
        # Gambar bounding box dari deteksi terakhir
        if self.last_detections:
            for det in self.last_detections:
                # Konversi koordinat ke integer
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                
                # Gambar bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Gambar label
                label = f"{det['name']}: {det['confidence']:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Tambahkan informasi FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def get_detection_results(self):
        """Ambil hasil deteksi terbaru"""
        try:
            return self.detection_queue.get_nowait()
        except queue.Empty:
            return None

# Muat model
model = load_model()

# Sidebar pengaturan
st.sidebar.title("⚙️ Pengaturan")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
detection_frequency = st.sidebar.slider("Frekuensi Deteksi (detik)", 0.1, 2.0, 0.5, 0.1)

# Pilihan sumber input
input_source = st.sidebar.radio(
    "Pilih Sumber Input", 
    ["Kamera WebRTC", "Upload Gambar", "Upload Video"]
)

# WebRTC Configuration (Updated for newer versions)
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Media constraints
media_stream_constraints = {
    "video": {
        "width": {"min": 640, "ideal": 1280}, 
        "height": {"min": 480, "ideal": 720}
    },
    "audio": False,
}

# Tambahkan informasi deployment
with st.sidebar.expander("ℹ️ Info Deployment"):
    st.write("""
    **Dependencies yang diperlukan:**
    ```
    pip install streamlit-webrtc opencv-python ultralytics pillow gdown
    ```
    
    **Untuk deployment:**
    1. Upload model ke Google Drive
    2. Set MODEL_URL di secrets/environment
    3. WebRTC bekerja di browser modern
    """)

# Kamera WebRTC
if input_source == "Kamera WebRTC":
    st.header("📹 Deteksi Real-time dengan WebRTC")
    
    # Inisialisasi transformer
    if 'webrtc_transformer' not in st.session_state:
        st.session_state.webrtc_transformer = DefectDetectionTransformer()
        if model:
            st.session_state.webrtc_transformer.set_model(model)
    
    # Update pengaturan transformer
    st.session_state.webrtc_transformer.set_conf_threshold(confidence_threshold)
    st.session_state.webrtc_transformer.set_detection_frequency(detection_frequency)
    
    # Kontrol WebRTC
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Pengaturan Kamera:**")
        show_stats = st.checkbox("Tampilkan Statistik Real-time", True)
    
    with col2:
        st.write("**Status Model:**")
        if model:
            st.success("✅ Siap")
        else:
            st.error("❌ Tidak tersedia")
    
    # WebRTC Streamer (Updated configuration)
    webrtc_ctx = webrtc_streamer(
        key="defect-detection",
        video_transformer_factory=lambda: st.session_state.webrtc_transformer,
        rtc_configuration=rtc_config,
        media_stream_constraints=media_stream_constraints,
        async_transform=True,
    )
    
    # Area untuk menampilkan hasil deteksi
    if webrtc_ctx.state.playing:
        # Placeholder untuk hasil deteksi dan statistik
        if show_stats:
            stats_placeholder = st.empty()
        detection_placeholder = st.empty()
        
        # Loop untuk menampilkan hasil deteksi
        while webrtc_ctx.state.playing:
            if st.session_state.webrtc_transformer:
                results = st.session_state.webrtc_transformer.get_detection_results()
                
                if results:
                    detections = results['detections']
                    fps = results['fps']
                    
                    # Tampilkan statistik
                    if show_stats:
                        with stats_placeholder.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("FPS", f"{fps:.1f}")
                            with col2:
                                st.metric("Defect Terdeteksi", len(detections))
                            with col3:
                                if detections:
                                    avg_conf = np.mean([d['confidence'] for d in detections])
                                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                                else:
                                    st.metric("Avg Confidence", "0.000")
                            with col4:
                                st.metric("Status", "🟢 Aktif")
                    
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
                            
                            # Alert untuk confidence tinggi
                            high_conf_detections = [d for d in detections if d['confidence'] > 0.8]
                            if high_conf_detections:
                                st.warning(f"⚠️ Ditemukan {len(high_conf_detections)} defect dengan confidence tinggi (>0.8)")
                        else:
                            st.success("✅ Tidak ada defect terdeteksi")
            
            time.sleep(0.1)  # Refresh setiap 100ms
    else:
        st.info("👆 Klik tombol 'START' untuk memulai deteksi real-time")
        st.write("""
        **Petunjuk Penggunaan WebRTC:**
        1. Klik tombol 'START' di atas
        2. Berikan izin akses kamera pada browser
        3. Tunggu beberapa detik untuk inisialisasi
        4. Arahkan kamera ke pakaian untuk mendeteksi defect
        5. Hasil deteksi akan ditampilkan di bawah video
        
        **Catatan:**
        - WebRTC bekerja optimal di browser Chrome/Firefox
        - Pastikan koneksi internet stabil
        - Gunakan pencahayaan yang cukup untuk hasil terbaik
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
                    
                    # Tambahkan tombol untuk deteksi batch
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔍 Deteksi Frame Ini"):
                            st.session_state['detect_single'] = True
                    with col2:
                        sample_frames = st.number_input("Deteksi Sample (frame)", min_value=1, max_value=min(50, frame_count), value=10)
                        if st.button("🎯 Deteksi Sample Frames"):
                            st.session_state['detect_sample'] = sample_frames
                    
                    # Deteksi single frame
                    if st.session_state.get('detect_single', False):
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
                        
                        st.session_state['detect_single'] = False
                    
                    # Deteksi sample frames
                    if st.session_state.get('detect_sample', 0) > 0:
                        sample_count = st.session_state['detect_sample']
                        
                        if model:
                            with st.spinner(f'Mendeteksi {sample_count} sample frames...'):
                                # Pilih frame secara merata
                                sample_indices = np.linspace(0, frame_count-1, sample_count, dtype=int)
                                
                                all_detections = []
                                progress_bar = st.progress(0)
                                
                                for i, frame_idx in enumerate(sample_indices):
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                    ret, frame = cap.read()
                                    
                                    if ret:
                                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        _, detections = detect_defects(model, frame_rgb, confidence_threshold)
                                        
                                        for det in detections:
                                            det['frame'] = frame_idx
                                            det['timestamp'] = frame_idx / fps
                                        
                                        all_detections.extend(detections)
                                    
                                    progress_bar.progress((i + 1) / sample_count)
                                
                                # Tampilkan hasil
                                if all_detections:
                                    st.subheader(f"📊 Hasil Deteksi {sample_count} Sample Frames")
                                    
                                    # Statistik
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Defect", len(all_detections))
                                    with col2:
                                        frames_with_defects = len(set(det['frame'] for det in all_detections))
                                        st.metric("Frames dengan Defect", frames_with_defects)
                                    with col3:
                                        avg_conf = np.mean([det['confidence'] for det in all_detections])
                                        st.metric("Avg Confidence", f"{avg_conf:.3f}")
                                    with col4:
                                        unique_classes = len(set(det['name'] for det in all_detections))
                                        st.metric("Jenis Defect", unique_classes)
                                    
                                    # Detail deteksi
                                    detection_df = []
                                    for det in all_detections:
                                        detection_df.append({
                                            "Frame": det['frame'],
                                            "Timestamp (s)": f"{det['timestamp']:.1f}",
                                            "Jenis": det['name'],
                                            "Confidence": f"{det['confidence']:.3f}",
                                            "Koordinat": f"({det['xmin']:.0f},{det['ymin']:.0f})-({det['xmax']:.0f},{det['ymax']:.0f})"
                                        })
                                    
                                    st.dataframe(detection_df, use_container_width=True)
                                else:
                                    st.success("✅ Tidak ada defect terdeteksi pada sample frames.")
                        
                        st.session_state['detect_sample'] = 0

                cap.release()
                
            except Exception as e:
                st.error(f"Error memproses video: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Petunjuk Penggunaan")
st.sidebar.markdown("""
**Kamera WebRTC:**
1. Pilih 'Kamera WebRTC'
2. Klik 'START' 
3. Berikan izin akses kamera
4. Lihat hasil deteksi real-time

**Dependencies:**
```
pip install streamlit-webrtc opencv-python ultralytics pillow gdown
```

**Keunggulan WebRTC:**
- Lebih stabil untuk streaming
- Kompatibel dengan deployment cloud
- Performa lebih baik
- Built-in error handling
""")

st.sidebar.markdown("---")
st.sidebar.write("© 2025 Deteksi Defect Pakaian")
st.sidebar.write("🌐 Powered by WebRTC!")

# Cleanup session state jika diperlukan
if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        if key.startswith(('uploaded_', 'video_', 'detect_')):
            del st.session_state[key]
    st.rerun()
