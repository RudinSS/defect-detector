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
import threading
import queue
import logging
import sys

# Suppress warnings and configure logging
logging.getLogger('aioice').setLevel(logging.CRITICAL)
logging.getLogger('aiortc').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.ERROR)

# Check if running in cloud environment
IS_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('STREAMLIT_CLOUD') or 'streamlit.io' in os.environ.get('HOST', '')

# Try to import streamlit-webrtc with better error handling
WEBRTC_AVAILABLE = False
if not IS_CLOUD:  # Only try WebRTC in local environments
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
        import av
        WEBRTC_AVAILABLE = True
    except ImportError:
        WEBRTC_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Deteksi Defect Pakaian",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .status-success { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>👕 Deteksi Defect Pakaian dengan YOLOv11</h1>
    <p>Aplikasi AI untuk mendeteksi cacat pada pakaian secara otomatis</p>
</div>
""", unsafe_allow_html=True)

# Model download and loading functions
@st.cache_resource(show_spinner=False)
def download_model():
    """Download model dari Google Drive atau URL eksternal"""
    model_url = os.getenv('MODEL_URL')
    model_path = "best.pt"
    
    if model_url:
        try:
            with st.spinner("🔄 Mengunduh model AI..."):
                if 'drive.google.com' in model_url:
                    file_id = model_url.split('/d/')[1].split('/')[0]
                    gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=True)
                else:
                    response = requests.get(model_url, timeout=30)
                    response.raise_for_status()
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                return model_path
        except Exception as e:
            st.error(f"❌ Gagal mengunduh model: {str(e)}")
            return None
    else:
        if not os.path.exists(model_path):
            st.warning("⚠️ Model tidak ditemukan. Set MODEL_URL di environment variables atau upload file model.")
        return model_path if os.path.exists(model_path) else None

@st.cache_resource(show_spinner=False)
def load_model():
    """Load YOLOv11 model dengan error handling"""
    try:
        model_path = download_model()
        if not model_path or not os.path.exists(model_path):
            return None
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            return model
        except ImportError:
            st.error("❌ Ultralytics tidak terinstall. Jalankan: `pip install ultralytics`")
            return None
            
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

def detect_defects(model, image, conf_threshold=0.5):
    """Deteksi defect pada gambar"""
    if model is None:
        return image, []
    
    try:
        results = model(image, conf=conf_threshold, verbose=False)
        annotated_image = results[0].plot()
        
        boxes = results[0].boxes
        names = results[0].names
        
        detections = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                box = boxes.xyxy[i].tolist()
                conf = boxes.conf[i].item()
                cls_id = int(boxes.cls[i].item())
                cls_name = names.get(cls_id, f"Class_{cls_id}")
                
                detections.append({
                    "Jenis Defect": cls_name,
                    "Confidence": round(conf, 3),
                    "X1": round(box[0], 1),
                    "Y1": round(box[1], 1),
                    "X2": round(box[2], 1),
                    "Y2": round(box[3], 1),
                    "Area": round((box[2] - box[0]) * (box[3] - box[1]), 1)
                })
        
        return annotated_image, detections
        
    except Exception as e:
        st.error(f"❌ Error dalam deteksi: {str(e)}")
        return image, []

# Enhanced VideoTransformer with better error handling
class DefectDetectionTransformer(VideoTransformerBase):
    def __init__(self, model=None, conf_threshold=0.5):
        self.model = model
        self.conf_threshold = conf_threshold
        self.frame_count = 0
        self.detection_count = 0
        self.last_detections = []
        self.process_every_n_frames = 5  # Process every 5th frame for performance
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Only process every nth frame to improve performance
            if self.frame_count % self.process_every_n_frames == 0 and self.model:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _, detections = detect_defects(self.model, img_rgb, self.conf_threshold)
                self.last_detections = detections
                if detections:
                    self.detection_count += len(detections)
            
            # Draw previous detections
            for det in self.last_detections:
                x1, y1, x2, y2 = int(det['X1']), int(det['Y1']), int(det['X2']), int(det['Y2'])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['Jenis Defect']}: {det['Confidence']:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add frame info
            cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Detections: {len(self.last_detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            self.frame_count += 1
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # Return original frame on error
            cv2.putText(img, f"Error: {str(e)[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# Load model
with st.spinner("🔄 Memuat model AI..."):
    model = load_model()

# Sidebar
st.sidebar.markdown("## ⚙️ Pengaturan")

# Model status
if model:
    st.sidebar.markdown('<p class="status-success">✅ Model siap digunakan</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<p class="status-error">❌ Model tidak tersedia</p>', unsafe_allow_html=True)
    st.sidebar.info("Upload model atau set MODEL_URL untuk menggunakan fitur deteksi.")

# Settings
confidence_threshold = st.sidebar.slider(
    "🎯 Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Tingkat kepercayaan minimum untuk deteksi"
)

# Input source selection
input_options = ["📁 Upload Gambar", "🎥 Upload Video"]
if WEBRTC_AVAILABLE and not IS_CLOUD:
    input_options.insert(0, "📹 Kamera Real-time")

input_source = st.sidebar.radio("📥 Pilih Sumber Input", input_options)

# Deployment info
with st.sidebar.expander("ℹ️ Info Deployment"):
    st.markdown("""
    **Dependencies:**
    ```bash
    pip install streamlit opencv-python ultralytics pillow gdown
    pip install streamlit-webrtc  # Untuk kamera real-time
    ```
    
    **Environment Variables:**
    - `MODEL_URL`: URL model YOLOv11 (.pt file)
    
    **Catatan:**
    - WebRTC hanya stabil di environment lokal
    - Untuk cloud deployment, gunakan upload file
    """)

# Main content based on input source
if "Kamera Real-time" in input_source and WEBRTC_AVAILABLE:
    st.header("📹 Deteksi Real-time")
    
    if IS_CLOUD:
        st.error("❌ WebRTC tidak didukung di Streamlit Cloud. Gunakan upload file sebagai alternatif.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("💡 **Tips:** Pastikan pencahayaan cukup dan kamera stabil untuk hasil optimal")
        
        with col2:
            if model:
                st.success("🟢 Siap mendeteksi")
            else:
                st.error("🔴 Model tidak tersedia")
        
        # WebRTC configuration
        rtc_config = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        })
        
        try:
            webrtc_ctx = webrtc_streamer(
                key="defect-detection-camera",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=lambda: DefectDetectionTransformer(model, confidence_threshold),
                rtc_configuration=rtc_config,
                media_stream_constraints={
                    "video": {"width": 640, "height": 480},
                    "audio": False
                }
            )
            
            if webrtc_ctx.state.playing:
                st.success("🔴 Live - Deteksi aktif")
            else:
                st.info("⏹️ Klik START untuk memulai deteksi real-time")
                
        except Exception as e:
            st.error(f"❌ Error WebRTC: {str(e)}")

elif "Upload Gambar" in input_source:
    st.header("📁 Analisis Gambar")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Pilih gambar pakaian:",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Format yang didukung: JPG, PNG, BMP, TIFF (Max: 10MB)"
    )
    
    if uploaded_file is not None:
        # File size check
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("❌ File terlalu besar! Maksimal 10MB.")
        else:
            try:
                # Process image
                image = Image.open(uploaded_file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize if too large
                max_size = 1920
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                img_array = np.array(image)
                
                # Display original image
                st.subheader("📷 Gambar Input")
                st.image(image, use_container_width=True)
                
                # Analyze button
                if st.button("🔍 Analisis Defect", type="primary", use_container_width=True):
                    if model:
                        with st.spinner("🔄 Menganalisis gambar..."):
                            img_with_boxes, detections = detect_defects(model, img_array, confidence_threshold)
                            
                            # Results
                            st.subheader("📊 Hasil Analisis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Gambar dengan Deteksi:**")
                                st.image(img_with_boxes, use_container_width=True)
                            
                            with col2:
                                if detections:
                                    # Statistics
                                    total_defects = len(detections)
                                    avg_confidence = np.mean([d['Confidence'] for d in detections])
                                    defect_types = len(set(d['Jenis Defect'] for d in detections))
                                    
                                    st.markdown("**Statistik:**")
                                    
                                    # Metrics
                                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                    with metrics_col1:
                                        st.metric("Total Defect", total_defects, delta=None)
                                    with metrics_col2:
                                        st.metric("Avg Confidence", f"{avg_confidence:.3f}", delta=None)
                                    with metrics_col3:
                                        st.metric("Jenis Defect", defect_types, delta=None)
                                    
                                    # Detailed results
                                    st.markdown("**Detail Deteksi:**")
                                    st.dataframe(detections, use_container_width=True)
                                    
                                    # Severity analysis
                                    high_conf = sum(1 for d in detections if d['Confidence'] > 0.8)
                                    if high_conf > 0:
                                        st.warning(f"⚠️ {high_conf} defect dengan confidence tinggi (>0.8)")
                                    
                                else:
                                    st.success("✅ Tidak ada defect terdeteksi!")
                                    st.balloons()
                    else:
                        st.error("❌ Model tidak tersedia untuk analisis.")
                        
            except Exception as e:
                st.error(f"❌ Error memproses gambar: {str(e)}")

elif "Upload Video" in input_source:
    st.header("🎥 Analisis Video")
    
    uploaded_video = st.file_uploader(
        "Pilih file video:",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Format yang didukung: MP4, AVI, MOV, MKV, WEBM (Max: 100MB)"
    )
    
    if uploaded_video is not None:
        if uploaded_video.size > 100 * 1024 * 1024:
            st.error("❌ File terlalu besar! Maksimal 100MB.")
        else:
            try:
                # Save video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_video.read())
                    video_path = tfile.name
                
                # Video info
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("❌ Gagal membuka video.")
                else:
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Video info display
                    st.subheader("📹 Informasi Video")
                    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                    
                    with info_col1:
                        st.metric("Durasi", f"{duration:.1f}s")
                    with info_col2:
                        st.metric("FPS", fps)
                    with info_col3:
                        st.metric("Resolusi", f"{width}x{height}")
                    with info_col4:
                        st.metric("Total Frame", frame_count)
                    
                    # Analysis options
                    st.subheader("⚙️ Opsi Analisis")
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        # Single frame analysis
                        st.markdown("**Analisis Frame Tunggal:**")
                        frame_number = st.slider('Pilih Frame', 0, frame_count-1, 0)
                        
                        if st.button("🔍 Analisis Frame Ini", use_container_width=True):
                            if model:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                                ret, frame = cap.read()
                                
                                if ret:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    
                                    with st.spinner("🔄 Menganalisis frame..."):
                                        frame_with_boxes, detections = detect_defects(model, frame_rgb, confidence_threshold)
                                        
                                        st.subheader(f"📊 Hasil Frame {frame_number}")
                                        
                                        display_col1, display_col2 = st.columns(2)
                                        
                                        with display_col1:
                                            st.markdown("**Frame Asli:**")
                                            st.image(frame_rgb, use_container_width=True)
                                        
                                        with display_col2:
                                            st.markdown("**Hasil Deteksi:**")
                                            st.image(frame_with_boxes, use_container_width=True)
                                        
                                        if detections:
                                            st.markdown("**Detail Deteksi:**")
                                            st.dataframe(detections, use_container_width=True)
                                        else:
                                            st.success("✅ Tidak ada defect pada frame ini")
                            else:
                                st.error("❌ Model tidak tersedia")
                    
                    with analysis_col2:
                        # Batch analysis
                        st.markdown("**Analisis Batch:**")
                        sample_frames = st.number_input(
                            "Jumlah sample frame:", 
                            min_value=5, 
                            max_value=min(50, frame_count), 
                            value=10
                        )
                        
                        if st.button("🎯 Analisis Sample Frames", use_container_width=True):
                            if model:
                                with st.spinner(f"🔄 Menganalisis {sample_frames} frames..."):
                                    # Sample frames evenly
                                    sample_indices = np.linspace(0, frame_count-1, sample_frames, dtype=int)
                                    
                                    all_detections = []
                                    frames_with_defects = 0
                                    progress_bar = st.progress(0)
                                    
                                    for i, frame_idx in enumerate(sample_indices):
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                        ret, frame = cap.read()
                                        
                                        if ret:
                                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                            _, detections = detect_defects(model, frame_rgb, confidence_threshold)
                                            
                                            if detections:
                                                frames_with_defects += 1
                                                for det in detections:
                                                    det['Frame'] = frame_idx
                                                    det['Timestamp'] = f"{frame_idx / fps:.1f}s"
                                                all_detections.extend(detections)
                                        
                                        progress_bar.progress((i + 1) / sample_frames)
                                    
                                    # Results summary
                                    st.subheader("📈 Ringkasan Analisis Batch")
                                    
                                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                                    
                                    with summary_col1:
                                        st.metric("Total Defect", len(all_detections))
                                    with summary_col2:
                                        st.metric("Frames dengan Defect", frames_with_defects)
                                    with summary_col3:
                                        if all_detections:
                                            avg_conf = np.mean([d['Confidence'] for d in all_detections])
                                            st.metric("Avg Confidence", f"{avg_conf:.3f}")
                                        else:
                                            st.metric("Avg Confidence", "0.000")
                                    with summary_col4:
                                        if all_detections:
                                            defect_types = len(set(d['Jenis Defect'] for d in all_detections))
                                            st.metric("Jenis Defect", defect_types)
                                        else:
                                            st.metric("Jenis Defect", "0")
                                    
                                    if all_detections:
                                        st.markdown("**Detail Semua Deteksi:**")
                                        st.dataframe(all_detections, use_container_width=True)
                                        
                                        # Defect timeline
                                        defect_by_frame = {}
                                        for det in all_detections:
                                            frame = det['Frame']
                                            if frame not in defect_by_frame:
                                                defect_by_frame[frame] = 0
                                            defect_by_frame[frame] += 1
                                        
                                        st.markdown("**Timeline Defect:**")
                                        timeline_data = []
                                        for frame, count in sorted(defect_by_frame.items()):
                                            timeline_data.append({
                                                'Frame': frame,
                                                'Timestamp': f"{frame/fps:.1f}s",
                                                'Jumlah Defect': count
                                            })
                                        st.dataframe(timeline_data, use_container_width=True)
                                    else:
                                        st.success("✅ Tidak ada defect terdeteksi di semua sample frames!")
                                        st.balloons()
                            else:
                                st.error("❌ Model tidak tersedia")
                    
                    cap.release()
                    
                    # Cleanup
                    try:
                        os.unlink(video_path)
                    except:
                        pass
                        
            except Exception as e:
                st.error(f"❌ Error memproses video: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🤖 Powered by YOLOv11 | 🚀 Built with Streamlit</p>
    <p>💡 Tips: Gunakan gambar dengan pencahayaan yang baik untuk hasil optimal</p>
</div>
""", unsafe_allow_html=True)
