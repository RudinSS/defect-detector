# CRITICAL FIX: Prevent PyTorch + Streamlit conflicts
import os
import warnings
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
warnings.filterwarnings("ignore")

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
from inference import get_model
import supervision as sv
import av
from PIL import Image
import io

# Additional PyTorch fix
import sys
import torch
if hasattr(torch, '_classes'):
    try:
        delattr(torch, '_classes')
    except:
        pass

# Page configuration
st.set_page_config(
    page_title="Clothing Defect Detection",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# WebRTC Connection Management
import atexit
import asyncio
import threading
from typing import Optional, Set
import weakref

# Global WebRTC connection tracker
_webrtc_connections: Set = weakref.WeakSet()
_cleanup_lock = threading.Lock()

def safe_cleanup_webrtc():
    """Safely cleanup WebRTC connections on app shutdown"""
    with _cleanup_lock:
        try:
            # Get current event loop or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Close all tracked connections
            for conn in list(_webrtc_connections):
                try:
                    if hasattr(conn, 'video_processor') and conn.video_processor:
                        conn.video_processor = None
                    if hasattr(conn, '_stop'):
                        conn._stop()
                except Exception as e:
                    print(f"Warning: Error closing WebRTC connection: {e}")
            
            # Clear the set
            _webrtc_connections.clear()
            
        except Exception as e:
            print(f"Warning: Error during WebRTC cleanup: {e}")

# Register cleanup
atexit.register(safe_cleanup_webrtc)

# Load YOLOv11 fine-tuned model from Roboflow
@st.cache_resource
def load_model():
    try:
        # Ganti dengan model ID Anda yang sebenarnya
        # Format: "workspace-name/project-name/version"
        model = get_model(model_id="deteksi-defect-pakaian-merges-fyiyz/1")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Pastikan model sudah benar")
        return None

# Load model at the top level
model = load_model()

# Initialize WebRTC session state
def init_webrtc_session():
    """Initialize WebRTC session state"""
    if 'webrtc_key' not in st.session_state:
        st.session_state.webrtc_key = "clothing-defect-detection"
    if 'webrtc_active' not in st.session_state:
        st.session_state.webrtc_active = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

init_webrtc_session()

# Helper function: Run detection and draw bounding boxes
def detect_and_annotate(image, confidence_threshold=0.5):
    """
    Detect objects in image and return annotated image
    """
    # Check if model is loaded
    if model is None:
        return image, None
        
    try:
        # Run inference
        results = model.infer(image, confidence=confidence_threshold)[0]
        
        # Convert to supervision detections
        detections = sv.Detections.from_inference(results)
        
        # Create annotators - using basic parameters for compatibility
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Annotate image
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        return annotated_image, detections
        
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return image, None

# Safe video frame callback wrapper
class SafeVideoProcessor:
    def __init__(self, model, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.error_count = 0
        self.max_errors = 10
        
    def __call__(self, frame):
        try:
            self.frame_count += 1
            
            # Reset error count periodically
            if self.frame_count % 100 == 0:
                self.error_count = 0
            
            # Skip processing if too many errors
            if self.error_count >= self.max_errors:
                return frame
            
            # Convert frame
            img = frame.to_ndarray(format="bgr24")
            
            # Run detection if model is available
            if self.model is not None:
                annotated_img, _ = detect_and_annotate(img, self.confidence_threshold)
                return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
            else:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
                
        except Exception as e:
            self.error_count += 1
            print(f"Video processing error #{self.error_count}: {e}")
            
            # Return original frame on error
            try:
                return frame
            except:
                # Create a black frame as fallback
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return av.VideoFrame.from_ndarray(black_frame, format="bgr24")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="Minimum confidence score for detections"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Model Info")
    if model:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model not loaded")
        st.warning("Check your model ID and API key")

# Main UI
st.title("👕 Clothing Defect Detection")
st.markdown("**Deteksi cacat pada pakaian menggunakan YOLOv11**")

if model is None:
    st.error("Model tidak dapat dimuat. Silakan periksa konfigurasi model Anda.")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Live Webcam"])

# Tab 1: Upload Image
with tab1:
    st.header("Upload Image Detection")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar pakaian...", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload image file untuk deteksi cacat pakaian"
    )
    
    if uploaded_file is not None:
        try:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Tidak dapat membaca file gambar. Pastikan format file didukung.")
            else:
                # Convert BGR to RGB for display
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image_rgb, caption="Original", use_container_width=True)
                
                with col2:
                    st.subheader("Detection Result")
                    with st.spinner("Melakukan deteksi..."):
                        annotated_image, detections = detect_and_annotate(image, confidence_threshold)
                        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_image_rgb, caption="Detection Result", use_container_width=True)
                
                # Show detection results
                if detections is not None and len(detections) > 0:
                    st.success(f"🔍 Ditemukan {len(detections)} deteksi")
                    
                    # Create results dataframe
                    results_data = []
                    for i, (bbox, confidence, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
                        results_data.append({
                            "Detection #": i+1,
                            "Class": detections.data.get('class_name', [f'Class_{class_id}'])[i] if 'class_name' in detections.data else f'Class_{class_id}',
                            "Confidence": f"{confidence:.2%}",
                            "Bounding Box": f"({int(bbox[0])}, {int(bbox[1])}) - ({int(bbox[2])}, {int(bbox[3])})"
                        })
                    
                    st.dataframe(results_data, use_container_width=True)
                else:
                    st.info("🔍 Tidak ada defect yang terdeteksi pada gambar ini")
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Tab 2: Live Webcam
# Tab 2: Live Webcam - Improved version
with tab2:
    st.header("Live Webcam Detection")
    st.markdown("**Real-time detection menggunakan webcam**")
    
    # WebRTC Configuration - Improved
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })
    
    # Instructions
    with st.expander("📋 Cara Penggunaan"):
        st.markdown("""
        1. Klik tombol **"START"** untuk memulai webcam
        2. Arahkan kamera ke pakaian yang ingin dideteksi
        3. Model akan mendeteksi defect secara real-time
        4. Klik **"STOP"** untuk menghentikan webcam
        
        **Tips:**
        - Pastikan pencahayaan cukup
        - Posisikan pakaian dengan jelas di depan kamera
        - Sesuaikan confidence threshold di sidebar jika diperlukan
        """)
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        if model is not None:
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Not Loaded")
    
    with col2:
        st.info(f"🎯 Confidence: {confidence_threshold}")
    
    with col3:
        frame_count = st.session_state.get('frame_count', 0)
        st.metric("Frames Processed", frame_count)
    
    # WebRTC Streamer with improved error handling
    if model is not None:
        try:
            # Create video processor
            video_processor = SafeVideoProcessor(model, confidence_threshold)
            
            # Create WebRTC streamer with unique key
            webrtc_ctx = webrtc_streamer(
                key=st.session_state.webrtc_key,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=video_processor,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1280, "max": 1920},
                        "height": {"min": 480, "ideal": 720, "max": 1080},
                        "frameRate": {"min": 15, "ideal": 30, "max": 30}
                    }, 
                    "audio": False
                },
                async_processing=True,
            )
            
            # Track connection for cleanup
            if webrtc_ctx:
                _webrtc_connections.add(webrtc_ctx)
                
                # Update session state
                st.session_state.webrtc_active = webrtc_ctx.state.playing
                if hasattr(video_processor, 'frame_count'):
                    st.session_state.frame_count = video_processor.frame_count
                
                # Status display
                if webrtc_ctx.state.playing:
                    st.success("🔴 **LIVE** - Detection Active")
                    
                    # Real-time stats
                    if hasattr(video_processor, 'error_count') and video_processor.error_count > 0:
                        st.warning(f"⚠️ Processing errors: {video_processor.error_count}")
                else:
                    st.info("⚪ Click **START** to begin detection")
            
        except Exception as e:
            st.error(f"❌ WebRTC Error: {str(e)}")
            st.info("💡 Try refreshing the page if the error persists")
            
            # Reset session state on error
            st.session_state.webrtc_active = False
            
            # Provide manual reset button
            if st.button("🔄 Reset WebRTC"):
                # Generate new key to force recreation
                import time
                st.session_state.webrtc_key = f"clothing-defect-detection-{int(time.time())}"
                st.experimental_rerun()
    else:
        st.error("❌ Cannot start webcam: Model not loaded")
        st.info("Please check your model configuration in the sidebar")
    
    # Emergency controls
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🛑 Emergency Stop"):
            try:
                safe_cleanup_webrtc()
                st.session_state.webrtc_active = False
                st.success("✅ WebRTC stopped successfully")
                time.sleep(1)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error during emergency stop: {e}")
    
    with col2:
        if st.button("🔄 Restart WebRTC"):
            try:
                safe_cleanup_webrtc()
                # Generate new key
                import time
                st.session_state.webrtc_key = f"clothing-defect-detection-{int(time.time())}"
                st.session_state.webrtc_active = False
                st.success("✅ WebRTC restarted")
                time.sleep(1)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error during restart: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 Powered by YOLOv11 & Roboflow | 🚀 Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
