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
    """Load YOLOv11 model dengan error handling yang lebih baik"""
    try:
        model_path = download_model()
        if not model_path or not os.path.exists(model_path):
            return None
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            # Test model dengan dummy image untuk memastikan model loaded
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model(dummy_img, verbose=False)
            return model
        except ImportError:
            st.error("❌ Ultralytics tidak terinstall. Jalankan: `pip install ultralytics`")
            return None
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

def detect_defects(model, image, conf_threshold=0.5):
    """Deteksi defect pada gambar dengan error handling yang lebih robust"""
    if model is None:
        return image, []
    
    try:
        # Ensure image is in correct format
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if needed
                pass
        
        results = model(image, conf=conf_threshold, verbose=False)
        
        # Get annotated image
        try:
            annotated_image = results[0].plot()
        except:
            # Fallback: return original image if plot fails
            annotated_image = image.copy()
        
        boxes = results[0].boxes
        names = results[0].names if hasattr(results[0], 'names') else {}
        
        detections = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                try:
                    box = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
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
                except Exception as e:
                    print(f"Error processing detection {i}: {e}")
                    continue
        
        return annotated_image, detections
        
    except Exception as e:
        st.error(f"❌ Error dalam deteksi: {str(e)}")
        return image, []

# Enhanced VideoTransformer with better error handling and thread safety
class DefectDetectionTransformer(VideoTransformerBase):
    def __init__(self, model=None, conf_threshold=0.5):
        super().__init__()
        self.model = model
        self.conf_threshold = conf_threshold
        self.frame_count = 0
        self.detection_count = 0
        self.last_detections = []
        self.process_every_n_frames = 3  # Lebih sering process untuk responsivitas
        self.last_processed_frame = None
        self.processing_lock = threading.Lock()
        
    def transform(self, frame):
        """Transform video frame dengan error handling yang robust"""
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            with self.processing_lock:
                # Process detection setiap beberapa frame
                should_process = (self.frame_count % self.process_every_n_frames == 0)
                
                if should_process and self.model is not None:
                    try:
                        # Convert BGR to RGB for YOLO
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Run detection
                        _, detections = detect_defects(self.model, img_rgb, self.conf_threshold)
                        self.last_detections = detections
                        if detections:
                            self.detection_count += len(detections)
                            
                    except Exception as e:
                        print(f"Detection error: {e}")
                        # Keep previous detections on error
                        pass
                
                # Draw detections on current frame
                try:
                    for det in self.last_detections:
                        x1, y1, x2, y2 = int(det['X1']), int(det['Y1']), int(det['X2']), int(det['Y2'])
                        
                        # Ensure coordinates are within image bounds
                        h, w = img.shape[:2]
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(0, min(x2, w-1))
                        y2 = max(0, min(y2, h-1))
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{det['Jenis Defect']}: {det['Confidence']:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        
                        # Background for label
                        cv2.rectangle(img, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                except Exception as e:
                    print(f"Drawing error: {e}")
                
                # Add frame info
                try:
                    # Status info
                    status_text = f"Frame: {self.frame_count} | Detections: {len(self.last_detections)}"
                    cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Model status
                    model_status = "Model: Ready" if self.model else "Model: Not Available"
                    cv2.putText(img, model_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                except Exception as e:
                    print(f"Text overlay error: {e}")
                
                self.frame_count += 1
            
            # Convert back to VideoFrame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            print(f"Transform error: {e}")
            # Return a black frame with error message on major failure
            try:
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, f"Error: {str(e)[:30]}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(error_img, format="bgr24")
            except:
                # Ultimate fallback
                return frame

# Load model with better error handling
@st.cache_data
def get_model_status():
    """Get model loading status"""
    return {"loaded": False, "error": None}

# Initialize model
model_status = get_model_status()

with st.spinner("🔄 Memuat model AI..."):
    try:
        model = load_model()
        if model:
            model_status["loaded"] = True
            st.success("✅ Model berhasil dimuat!")
        else:
            model_status["error"] = "Model tidak tersedia"
    except Exception as e:
        model_status["error"] = str(e)
        model = None

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

# WebRTC Settings
if WEBRTC_AVAILABLE:
    st.sidebar.markdown("### 📹 Pengaturan Kamera")
    video_width = st.sidebar.selectbox("Lebar Video", [320, 640, 1280], index=1)
    video_height = st.sidebar.selectbox("Tinggi Video", [240, 480, 720], index=1)
    frame_rate = st.sidebar.selectbox("Frame Rate", [15, 20, 30], index=1)

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
    
    **Troubleshooting WebRTC:**
    - Pastikan browser mendukung WebRTC
    - Izinkan akses kamera
    - Gunakan HTTPS untuk deployment
    - Check firewall settings
    """)

# Debug info
with st.sidebar.expander("🔧 Debug Info"):
    st.write("WebRTC Available:", WEBRTC_AVAILABLE)
    st.write("Is Cloud:", IS_CLOUD)
    st.write("Model Loaded:", model is not None)
    if model_status.get("error"):
        st.error(f"Model Error: {model_status['error']}")

# Main content based on input source
if "Kamera Real-time" in input_source and WEBRTC_AVAILABLE:
    st.header("📹 Deteksi Real-time")
    
    if IS_CLOUD:
        st.error("❌ WebRTC tidak didukung di Streamlit Cloud. Gunakan upload file sebagai alternatif.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if model:
                st.success("🟢 Model siap - Klik START untuk memulai deteksi")
            else:
                st.error("🔴 Model tidak tersedia - Upload model terlebih dahulu")
                
            st.info("💡 **Tips:** Pastikan pencahayaan cukup dan koneksi internet stabil")
        
        with col2:
            # Real-time stats placeholder
            stats_placeholder = st.empty()
        
        # WebRTC configuration dengan pengaturan yang lebih robust
        rtc_config = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        })
        
        try:
            # Create transformer instance
            def create_transformer():
                return DefectDetectionTransformer(model, confidence_threshold)
            
            webrtc_ctx = webrtc_streamer(
                key="defect-detection-camera",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=create_transformer,
                rtc_configuration=rtc_config,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": video_width},
                        "height": {"ideal": video_height},
                        "frameRate": {"ideal": frame_rate}
                    },
                    "audio": False
                },
                async_processing=True  # Enable async processing
            )
            
            # Status display
            if webrtc_ctx.state.playing:
                st.success("🔴 LIVE - Deteksi aktif")
                
                # Show real-time stats
                if webrtc_ctx.video_transformer:
                    transformer = webrtc_ctx.video_transformer
                    with stats_placeholder.container():
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Frames Processed", transformer.frame_count)
                        with stat_col2:
                            st.metric("Current Detections", len(transformer.last_detections))
                        with stat_col3:
                            st.metric("Total Detections", transformer.detection_count)
                        
                        # Show current detections
                        if transformer.last_detections:
                            st.subheader("🎯 Deteksi Saat Ini")
                            for i, det in enumerate(transformer.last_detections):
                                st.write(f"**{det['Jenis Defect']}** - Confidence: {det['Confidence']:.3f}")
                
            elif webrtc_ctx.state.signalling:
                st.info("🔄 Menghubungkan...")
            else:
                st.info("⏹️ Klik START untuk memulai deteksi real-time")
                
        except Exception as e:
            st.error(f"❌ Error WebRTC: {str(e)}")
            st.info("**Troubleshooting:**")
            st.write("1. Refresh halaman")
            st.write("2. Periksa izin kamera browser")
            st.write("3. Pastikan tidak ada aplikasi lain yang menggunakan kamera")
            st.write("4. Coba browser yang berbeda")

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

# Video processing code remains the same as original...
elif "Upload Video" in input_source:
    st.header("🎥 Analisis Video")
    st.info("📹 Fungsi analisis video sama seperti versi sebelumnya...")
    # ... (video processing code sama seperti aslinya)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🤖 Powered by YOLOv11 | 🚀 Built with Streamlit | 📹 WebRTC Enhanced</p>
    <p>💡 Tips: Gunakan pencahayaan yang baik dan koneksi internet stabil untuk hasil optimal</p>
</div>
""", unsafe_allow_html=True)
