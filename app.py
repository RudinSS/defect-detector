import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import os
import gdown
import tempfile
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO
import logging
from PIL import Image
import io

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Defect Pakaian",
    page_icon="👔",
    layout="wide"
)

class YOLOv11DefectDetector(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.5
        self.load_model()
    
    def load_model(self):
        """Load YOLOv11 model dari Google Drive atau lokal"""
        try:
            model_path = self.get_model_path()
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info(f"Model berhasil dimuat dari: {model_path}")
            else:
                # Fallback ke model default YOLOv11
                self.model = YOLO('yolo11n.pt')
                logger.warning("Menggunakan model YOLOv11 default")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def get_model_path(self):
        """Download model dari Google Drive jika diperlukan"""
        try:
            # Cek environment variable untuk URL model
            model_url = os.getenv('MODEL_URL')
            
            if model_url:
                # Konversi Google Drive sharing URL ke direct download URL
                if 'drive.google.com' in model_url:
                    file_id = model_url.split('/d/')[1].split('/')[0]
                    download_url = f'https://drive.google.com/uc?id={file_id}'
                else:
                    download_url = model_url
                
                # Path untuk menyimpan model
                model_path = "yolov11_defect_model.pt"
                
                # Download jika belum ada
                if not os.path.exists(model_path):
                    with st.spinner('Mengunduh model...'):
                        gdown.download(download_url, model_path, quiet=False)
                        logger.info(f"Model berhasil diunduh ke: {model_path}")
                
                return model_path
            else:
                # Cek model lokal
                local_paths = [
                    "yolov11_defect_model.pt",
                    "models/yolov11_defect_model.pt",
                    "./best.pt"
                ]
                
                for path in local_paths:
                    if os.path.exists(path):
                        return path
                
                return None
                
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return None
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold untuk deteksi"""
        self.confidence_threshold = threshold
    
    def transform(self, frame):
        """Transform frame untuk deteksi real-time"""
        img = frame.to_ndarray(format="bgr24")
        
        if self.model is None:
            # Tampilkan pesan error di frame
            cv2.putText(img, "Model tidak tersedia", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img
        
        return self.detect_and_draw(img)
    
    def detect_and_draw(self, img):
        """Fungsi deteksi dan draw bounding box yang dapat digunakan untuk video dan gambar"""
        try:
            # Prediksi menggunakan YOLOv11
            results = self.model(img, conf=self.confidence_threshold, verbose=False)
            
            # Debug: tampilkan info deteksi
            detection_count = 0
            
            # Draw bounding boxes dan labels
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    detection_count = len(boxes)
                    
                    for i, box in enumerate(boxes):
                        try:
                            # Koordinat bounding box - pastikan dalam format yang benar
                            coords = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = coords.astype(int)
                            
                            # Pastikan koordinat valid
                            h, w = img.shape[:2]
                            x1 = max(0, min(x1, w-1))
                            y1 = max(0, min(y1, h-1))
                            x2 = max(x1+1, min(x2, w))
                            y2 = max(y1+1, min(y2, h))
                            
                            # Confidence score
                            conf = float(box.conf[0].cpu().numpy())
                            
                            # Class name
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names.get(cls, f"Class_{cls}")
                            
                            # Warna berdasarkan jenis defect
                            color = self.get_defect_color(class_name)
                            
                            # Draw bounding box dengan thickness yang lebih tebal
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                            
                            # Label dengan confidence
                            label = f"{class_name}: {conf:.3f}"
                            
                            # Setup text
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.7
                            thickness = 2
                            (text_width, text_height), baseline = cv2.getTextSize(
                                label, font, font_scale, thickness
                            )
                            
                            # Background untuk label
                            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
                            cv2.rectangle(
                                img, 
                                (x1, label_y - text_height - 5), 
                                (x1 + text_width, label_y + 5), 
                                color, 
                                -1
                            )
                            
                            # Text label
                            cv2.putText(
                                img, label, (x1, label_y), 
                                font, font_scale, (255, 255, 255), thickness
                            )
                            
                        except Exception as box_error:
                            logger.error(f"Error processing box {i}: {box_error}")
                            continue
            
            # Tampilkan info deteksi di pojok kiri atas
            info_text = f"Detections: {detection_count} | Conf: {self.confidence_threshold:.2f}"
            cv2.putText(img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Tampilkan status model
            model_status = "Custom Model" if os.getenv('MODEL_URL') else "Default YOLOv11"
            cv2.putText(img, f"Model: {model_status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            logger.error(f"Error dalam deteksi: {e}")
            cv2.putText(img, f"Detection Error: {str(e)[:50]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return img
    
    def get_defect_color(self, class_name):
        """Mendapatkan warna berdasarkan jenis defect"""
        color_map = {
            'stain': (0, 0, 255),      # Merah untuk noda
            'hole': (255, 0, 0),       # Biru untuk lubang
            'tear': (0, 255, 255),     # Kuning untuk robek
            'fade': (255, 0, 255),     # Magenta untuk pudar
            'wrinkle': (0, 255, 0),    # Hijau untuk kerut
            'default': (255, 255, 0)   # Cyan untuk lainnya
        }
        
        return color_map.get(class_name.lower(), color_map['default'])

def detect_image_defects(image, model, confidence_threshold=0.5):
    """Fungsi untuk deteksi defect pada gambar statis"""
    if model is None:
        return None, "Model tidak tersedia"
    
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            if img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
            img_array = image
        
        # Prediksi
        results = model(img_array, conf=confidence_threshold, verbose=False)
        
        # Convert ke BGR untuk OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        detection_results = []
        
        # Draw bounding boxes
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Koordinat bounding box
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords.astype(int)
                    
                    # Confidence score
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Class name
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names.get(cls, f"Class_{cls}")
                    
                    # Simpan hasil deteksi
                    detection_results.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    # Warna berdasarkan jenis defect
                    color_map = {
                        'stain': (0, 0, 255),      # Merah
                        'hole': (255, 0, 0),       # Biru
                        'tear': (0, 255, 255),     # Kuning
                        'fade': (255, 0, 255),     # Magenta
                        'wrinkle': (0, 255, 0),    # Hijau
                        'default': (255, 255, 0)   # Cyan
                    }
                    
                    color = color_map.get(class_name.lower(), color_map['default'])
                    
                    # Draw bounding box
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
                    
                    # Label dengan confidence
                    label = f"{class_name}: {conf:.3f}"
                    
                    # Setup text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Background untuk label
                    label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
                    cv2.rectangle(
                        img_bgr, 
                        (x1, label_y - text_height - 5), 
                        (x1 + text_width, label_y + 5), 
                        color, 
                        -1
                    )
                    
                    # Text label
                    cv2.putText(
                        img_bgr, label, (x1, label_y), 
                        font, font_scale, (255, 255, 255), thickness
                    )
        
        # Convert kembali ke RGB untuk display
        result_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return result_image, detection_results
        
    except Exception as e:
        logger.error(f"Error dalam deteksi gambar: {e}")
        return None, f"Error: {str(e)}"

def load_model_for_image():
    """Load model untuk deteksi gambar"""
    try:
        # Cek environment variable untuk URL model
        model_url = os.getenv('MODEL_URL')
        
        if model_url:
            # Konversi Google Drive sharing URL ke direct download URL
            if 'drive.google.com' in model_url:
                file_id = model_url.split('/d/')[1].split('/')[0]
                download_url = f'https://drive.google.com/uc?id={file_id}'
            else:
                download_url = model_url
            
            # Path untuk menyimpan model
            model_path = "yolov11_defect_model.pt"
            
            # Download jika belum ada
            if not os.path.exists(model_path):
                with st.spinner('Mengunduh model...'):
                    gdown.download(download_url, model_path, quiet=False)
                    logger.info(f"Model berhasil diunduh ke: {model_path}")
            
            return YOLO(model_path)
        else:
            # Cek model lokal
            local_paths = [
                "yolov11_defect_model.pt",
                "models/yolov11_defect_model.pt",
                "./best.pt"
            ]
            
            for path in local_paths:
                if os.path.exists(path):
                    return YOLO(path)
            
            # Fallback ke model default
            return YOLO('yolo11n.pt')
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def main():
    st.title("🔍 Deteksi Defect Pakaian Real-time")
    st.markdown("---")
    
    # Tabs untuk berbagai mode deteksi
    tab1, tab2 = st.tabs(["📹 Real-time Camera", "🖼️ Upload Gambar"])
    
    # Sidebar untuk konfigurasi
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # Confidence threshold
        confidence = st.slider(
            "Confidence Threshold", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.1, 
            step=0.01,
            help="Ambang batas kepercayaan untuk deteksi (lebih rendah = lebih sensitif)"
        )
        
        # Model status
        st.header("📊 Status Model")
        model_url = os.getenv('MODEL_URL', 'Tidak ada')
        if model_url != 'Tidak ada':
            st.success("✅ Custom model configured")
        else:
            st.warning("⚠️ Using default YOLOv11")
        
        # Informasi defect yang dapat dideteksi
        st.header("🏷️ Jenis Defect")
        defect_info = {
            "🔴 Noda (Stain)": "Kotoran atau bercak pada kain",
            "🔵 Lubang (Hole)": "Bolong atau berlubang", 
            "🟡 Robek (Tear)": "Sobek atau koyak",
            "🟣 Pudar (Fade)": "Warna memudar atau belang",
            "🟢 Kerut (Wrinkle)": "Kusut atau berkerut"
        }
        
        for defect, desc in defect_info.items():
            st.write(f"{defect}")
            st.caption(desc)
    
    # Tab 1: Real-time Camera Detection
    with tab1:
        st.header("📹 Deteksi Real-time")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Konfigurasi WebRTC
            RTC_CONFIGURATION = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            })
            
            # WebRTC Streamer
            webrtc_ctx = webrtc_streamer(
                key="defect-detection",
                video_transformer_factory=YOLOv11DefectDetector,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1280, "max": 1920},
                        "height": {"min": 480, "ideal": 720, "max": 1080},
                    },
                    "audio": False
                },
                async_processing=True,
            )
            
            # Update confidence threshold
            if webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.set_confidence_threshold(confidence)
        
        with col2:
            st.header("📋 Status")
            
            # Status connection
            if webrtc_ctx.state.playing:
                st.success("🟢 Kamera aktif")
                st.info("🔍 Deteksi berjalan")
            else:
                st.warning("🟡 Kamera tidak aktif")
                st.info("👆 Klik START untuk mulai")
            
            # Current settings
            st.header("⚙️ Pengaturan Aktif")
            st.metric("Confidence", f"{confidence:.2f}")
            st.metric("Model Status", "Loaded" if webrtc_ctx.video_transformer else "Loading")
    
    # Tab 2: Image Upload Detection
    with tab2:
        st.header("🖼️ Upload dan Deteksi Gambar")
        
        # Upload gambar
        uploaded_files = st.file_uploader(
            "Pilih gambar pakaian untuk dianalisis",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload gambar dalam format PNG, JPG, atau JPEG"
        )
        
        if uploaded_files:
            # Load model untuk deteksi gambar
            if 'image_model' not in st.session_state:
                with st.spinner('Loading model untuk deteksi gambar...'):
                    st.session_state.image_model = load_model_for_image()
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"### 📸 Gambar {i+1}: {uploaded_file.name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🖼️ Gambar Asli")
                    
                    # Load dan display gambar asli
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Gambar Original", use_column_width=True)
                    
                    # Info gambar
                    st.info(f"📏 Ukuran: {image.size[0]} x {image.size[1]} pixels")
                
                with col2:
                    st.subheader("🔍 Hasil Deteksi")
                    
                    if st.session_state.image_model is not None:
                        # Proses deteksi
                        with st.spinner('Memproses deteksi...'):
                            result_image, detection_results = detect_image_defects(
                                image, 
                                st.session_state.image_model, 
                                confidence
                            )
                        
                        if result_image is not None:
                            # Display hasil
                            st.image(result_image, caption="Hasil Deteksi", use_column_width=True)
                            
                            # Summary hasil deteksi
                            if isinstance(detection_results, list) and len(detection_results) > 0:
                                st.success(f"✅ Ditemukan {len(detection_results)} defect:")
                                
                                # Tabel hasil deteksi
                                for j, detection in enumerate(detection_results):
                                    st.write(f"**{j+1}. {detection['class'].title()}**")
                                    st.write(f"   - Confidence: {detection['confidence']:.3f}")
                                    st.write(f"   - Posisi: {detection['bbox']}")
                                
                                # Chart statistik defect
                                defect_counts = {}
                                for detection in detection_results:
                                    defect_type = detection['class']
                                    defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
                                
                                st.bar_chart(defect_counts)
                                
                            else:
                                st.info("ℹ️ Tidak ada defect terdeteksi pada threshold ini")
                                st.caption("Coba turunkan confidence threshold jika diperlukan")
                        else:
                            st.error(f"❌ Error: {detection_results}")
                    else:
                        st.error("❌ Model tidak dapat dimuat")
                
                st.markdown("---")
    
    # Footer information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model", "YOLOv11", "Custom Trained")
    
    with col2:
        st.metric("Framework", "Ultralytics", "Latest")
    
    with col3:
        st.metric("Deployment", "Streamlit", "WebRTC + Upload")

if __name__ == "__main__":
    main()
