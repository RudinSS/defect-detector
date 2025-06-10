import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
from inference import get_model
import supervision as sv
import av
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Clothing Defect Detection",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load YOLOv11 fine-tuned model from Roboflow
@st.cache_resource
def load_model():
    try:
        # Ganti dengan model ID Anda yang sebenarnya
        # Format: "workspace-name/project-name/version"
        model = get_model(model_id="deteksi-defect-pakaian-merges/1")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Pastikan model sudah benar")
        return None

# Helper function: Run detection and draw bounding boxes
def detect_and_annotate(image, confidence_threshold=0.5):
    """
    Detect objects in image and return annotated image
    """
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

# Load model
model = load_model()

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
st.title("👗 Clothing Defect Detection")
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
with tab2:
    st.header("Live Webcam Detection")
    st.markdown("**Real-time detection menggunakan webcam**")
    
    # WebRTC Configuration
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
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
    
    # WebRTC Streamer
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Run detection
        annotated_img, _ = detect_and_annotate(img, confidence_threshold)
        
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
    
    webrtc_ctx = webrtc_streamer(
        key="clothing-defect-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 Powered by YOLOv11 & Roboflow | 🚀 Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
