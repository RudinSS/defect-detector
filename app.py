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

# Helper function: Run detection and draw bounding boxes with confidence
def detect_and_annotate(image, confidence_threshold=0.5):
    """
    Detect objects in image and return annotated image with confidence scores
    """
    # Check if model is loaded
    if model is None:
        return image, None
        
    try:
        # Debug: Print image info
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        
        # Run inference with lower confidence to catch more detections
        results = model.infer(image, confidence=max(0.01, confidence_threshold - 0.2))[0]
        
        # Debug: Print raw results
        print(f"Raw results type: {type(results)}")
        if hasattr(results, 'predictions'):
            print(f"Number of predictions: {len(results.predictions) if results.predictions else 0}")
            if results.predictions:
                for i, pred in enumerate(results.predictions):
                    print(f"Prediction {i}: {pred}")
        
        # Convert to supervision detections
        detections = sv.Detections.from_inference(results)
        
        # Debug: Print detections info
        print(f"Number of detections: {len(detections)}")
        if len(detections) > 0:
            print(f"Detection confidences: {detections.confidence}")
            print(f"Detection classes: {detections.class_id}")
        
        # Filter detections by user-specified confidence threshold
        if len(detections) > 0:
            mask = detections.confidence >= confidence_threshold
            detections = detections[mask]
        
        # Create annotators
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Create custom labels with confidence scores
        labels = []
        if len(detections) > 0:
            for i, (confidence, class_id) in enumerate(zip(detections.confidence, detections.class_id)):
                # Get class name if available, otherwise use class_id
                if hasattr(results, 'predictions') and results.predictions:
                    # Find the corresponding prediction for this detection
                    class_name = f'Class_{class_id}'
                    for pred in results.predictions:
                        if hasattr(pred, 'class_id') and pred.class_id == class_id:
                            if hasattr(pred, 'class_name') or hasattr(pred, 'class'):
                                class_name = getattr(pred, 'class_name', getattr(pred, 'class', f'Class_{class_id}'))
                                break
                else:
                    class_name = f'Class_{class_id}'
                
                # Create label with class name and confidence percentage
                label = f"{class_name} {confidence:.1%}"
                labels.append(label)
        
        # Annotate image with bounding boxes
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        
        # Annotate image with custom labels (including confidence)
        if len(detections) > 0:
            annotated_image = label_annotator.annotate(
                scene=annotated_image, 
                detections=detections,
                labels=labels
            )
        
        return annotated_image, detections
        
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return image, None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.25, 
        step=0.01,
        help="Minimum confidence score for detections. Try lowering this if no detections appear."
    )
    
    # Debug options
    st.markdown("### 🔧 Debug Options")
    show_debug = st.checkbox("Show Debug Information", value=False)
    
    # Model troubleshooting
    st.markdown("### 🩺 Troubleshoot")
    if st.button("Test Model"):
        if model:
            try:
                # Create a test image
                test_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
                test_results = model.infer(test_img, confidence=0.01)[0]
                st.success("✅ Model is working")
                st.info(f"Test result type: {type(test_results)}")
            except Exception as e:
                st.error(f"❌ Model test failed: {e}")
        else:
            st.error("❌ Model not loaded")
    
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
                        
                        # Show debug info if enabled
                        if show_debug:
                            st.markdown("**Debug Information:**")
                            st.text(f"Image shape: {image.shape}")
                            st.text(f"Confidence threshold: {confidence_threshold}")
                
                # Show detection results
                if detections is not None and len(detections) > 0:
                    st.success(f"🔍 Ditemukan {len(detections)} deteksi")
                    
                    # Create results dataframe with enhanced information
                    results_data = []
                    for i, (bbox, confidence, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
                        # Try to get class name from model results
                        try:
                            results = model.infer(image, confidence=0.01)[0]
                            class_name = f'Class_{class_id}'
                            if hasattr(results, 'predictions') and results.predictions:
                                for pred in results.predictions:
                                    if hasattr(pred, 'class_id') and pred.class_id == class_id:
                                        if hasattr(pred, 'class_name') or hasattr(pred, 'class'):
                                            class_name = getattr(pred, 'class_name', getattr(pred, 'class', f'Class_{class_id}'))
                                            break
                        except:
                            class_name = f'Class_{class_id}'
                        
                        results_data.append({
                            "Detection #": i+1,
                            "Class": class_name,
                            "Confidence": f"{confidence:.2%}",
                            "Confidence Score": f"{confidence:.4f}",
                            "Bounding Box": f"({int(bbox[0])}, {int(bbox[1])}) - ({int(bbox[2])}, {int(bbox[3])})"
                        })
                    
                    st.dataframe(results_data, use_container_width=True)
                else:
                    st.warning("🔍 Tidak ada defect yang terdeteksi pada gambar ini")
                    st.info("💡 **Tips untuk meningkatkan deteksi:**")
                    st.markdown("""
                    - Coba turunkan **Confidence Threshold** di sidebar (misal ke 0.1 atau 0.05)
                    - Pastikan model sudah dilatih dengan data yang sesuai
                    - Periksa apakah gambar memiliki pencahayaan yang baik
                    - Aktifkan **Debug Information** untuk melihat detail lebih lanjut
                    """)
                    
                    # Suggest trying different confidence levels
                    st.markdown("**🎯 Coba confidence threshold yang berbeda:**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("Try 0.1"):
                            st.session_state.suggested_confidence = 0.1
                    with col_b:
                        if st.button("Try 0.05"):
                            st.session_state.suggested_confidence = 0.05
                    with col_c:
                        if st.button("Try 0.01"):
                            st.session_state.suggested_confidence = 0.01
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Tab 2: Live Webcam
with tab2:
    st.header("Live Webcam Detection")
    st.markdown("**Real-time detection menggunakan webcam**")
    
    # WebRTC Configuration - Fixed deprecated parameter
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # Instructions
    with st.expander("📋 Cara Penggunaan"):
        st.markdown("""
        1. Klik tombol **"START"** untuk memulai webcam
        2. Arahkan kamera ke pakaian yang ingin dideteksi
        3. Model akan mendeteksi defect secara real-time dengan confidence score
        4. Klik **"STOP"** untuk menghentikan webcam
        
        **Tips:**
        - Pastikan pencahayaan cukup
        - Posisikan pakaian dengan jelas di depan kamera
        - Sesuaikan confidence threshold di sidebar jika diperlukan
        - Label akan menampilkan nama class dan confidence score (contoh: "Defect 85.3%")
        """)
    
    # WebRTC Streamer with proper error handling
    def video_frame_callback(frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Run detection only if model is available
            if model is not None:
                annotated_img, _ = detect_and_annotate(img, confidence_threshold)
                return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
            else:
                # Return original frame if model is not available
                return av.VideoFrame.from_ndarray(img, format="bgr24")
                
        except Exception as e:
            # Log error and return original frame
            print(f"Error in video callback: {e}")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # Only create webrtc_streamer if model is loaded
    if model is not None:
        webrtc_ctx = webrtc_streamer(
            key="clothing-defect-detection",
            mode=WebRtcMode.SENDRECV,
            # Use the new parameter names instead of deprecated rtc_configuration
            frontend_rtc_configuration=RTC_CONFIGURATION,
            server_rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Display current confidence threshold for webcam
        st.info(f"🎯 Current confidence threshold: {confidence_threshold:.1%}")
    else:
        st.error("Cannot start webcam: Model not loaded")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 Powered by YOLOv11 & Roboflow | 🚀 Built with Streamlit</p>
    <p><small>📊 Detection labels now include confidence scores for better analysis</small></p>
</div>
""", unsafe_allow_html=True)
