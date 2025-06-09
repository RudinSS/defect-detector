import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
import tempfile
from ultralytics import YOLO

# Judul aplikasi
st.title('Deteksi Defect Pakaian dengan YOLOv11')
st.write('Aplikasi ini mendeteksi defect pada pakaian menggunakan model YOLOv11')

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    model_path = "runs/detect/train52/weights/best.pt"  # Ganti dengan path model kamu
    try:
        model = YOLO(model_path)
        st.success("Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Fungsi deteksi
def detect_defects(model, image, conf_threshold):
    results = model(image, conf=conf_threshold)  # Menerapkan threshold langsung pada model
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

# Muat model
model = load_model()

# Sidebar pengaturan
st.sidebar.title("Pengaturan")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
input_source = st.sidebar.radio("Pilih Sumber Input", ["Kamera", "Upload Gambar", "Upload Video"])

# Fungsi untuk mendapatkan daftar kamera yang tersedia
def get_available_cameras(max_cameras=10):
    """Mendapatkan daftar kamera yang tersedia"""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Coba dapatkan properti kamera
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Nama kamera (gunakan backend jika tersedia)
            camera_name = f"Kamera {i}"
            if hasattr(cap, 'getBackendName'):
                try:
                    backend = cap.getBackendName()
                    camera_name = f"Kamera {i} ({backend})"
                except:
                    pass
            
            camera_info = {
                "index": i,
                "name": camera_name,
                "resolution": f"{width}x{height}",
                "fps": fps
            }
            available_cameras.append(camera_info)
        cap.release()
    
    return available_cameras

# Deteksi via Kamera
if input_source == "Kamera":
    st.header("Deteksi via Kamera")
    
    # Dapatkan daftar kamera yang tersedia
    with st.spinner("Mencari kamera yang tersedia..."):
        available_cameras = get_available_cameras()
    
    # Buat opsi untuk dropdown pemilihan kamera
    camera_options = []
    for cam in available_cameras:
        option_text = f"{cam['name']} - {cam['resolution']} @ {cam['fps']} FPS"
        camera_options.append((option_text, cam['index']))
    
    # Jika tidak ada kamera yang terdeteksi
    if not camera_options:
        st.warning("Tidak ada kamera yang terdeteksi.")
        camera_options = [("Kamera Default (0)", 0)]
    
    # Dropdown untuk memilih kamera
    selected_option = st.selectbox(
        "Pilih Kamera:",
        options=[option[0] for option in camera_options],
        index=0
    )
    
    # Dapatkan indeks kamera dari opsi yang dipilih
    selected_camera_index = 0
    for option, index in camera_options:
        if option == selected_option:
            selected_camera_index = index
            break
    
    # Widget untuk memulai/menghentikan kamera
    col1, col2 = st.columns(2)
    with col1:
        run = st.button('Mulai Kamera')
    with col2:
        stop = st.button('Hentikan Kamera')
    
    # Frame untuk menampilkan video
    FRAME_WINDOW = st.image([])
    
    if run:
        st.info(f"Membuka kamera dengan indeks: {selected_camera_index}")
        cap = cv2.VideoCapture(selected_camera_index)
        
        if not cap.isOpened():
            st.error(f"Tidak dapat mengakses kamera dengan indeks {selected_camera_index}. Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
        else:
            st.session_state['camera_running'] = True
            
            # Tampilkan informasi kamera
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            st.success(f"Kamera terhubung: {width}x{height} @ {fps} FPS")
            
            while st.session_state.get('camera_running', True):
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal mendapatkan frame dari kamera.")
                    break
                
                # Konversi BGR ke RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if model:
                    # Deteksi defect
                    frame_with_boxes, _ = detect_defects(model, frame_rgb, confidence_threshold)
                    
                    # Tampilkan deteksi (tanpa log terdeteksi di bawah)
                    FRAME_WINDOW.image(frame_with_boxes)
                else:
                    FRAME_WINDOW.image(frame_rgb)
                
                time.sleep(0.1)  # Sedikit delay untuk mengurangi beban CPU
                
                if stop:
                    st.session_state['camera_running'] = False
                    break
            
            cap.release()

# Deteksi via Gambar
elif input_source == "Upload Gambar":
    st.header("Deteksi via Upload Gambar")
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Simpan gambar ke session state agar bisa digunakan kembali saat threshold berubah
        if 'uploaded_image' not in st.session_state or st.session_state['uploaded_image_name'] != uploaded_file.name:
            image = Image.open(uploaded_file)
            st.session_state['uploaded_image'] = np.array(image)
            st.session_state['uploaded_image_name'] = uploaded_file.name
            
        img_array = st.session_state['uploaded_image']
        
        if model:
            with st.spinner('Mendeteksi defect...'):
                # Gunakan threshold saat ini
                img_with_boxes, detections = detect_defects(model, img_array, confidence_threshold)
                
                # Tampilkan gambar asli dan hasil deteksi secara bersampingan
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Gambar Asli")
                    st.image(img_array, channels="RGB", use_container_width=True)
                
                with col2:
                    st.subheader("Hasil Deteksi")
                    st.image(img_with_boxes, channels="RGB", use_container_width=True)
                
                # Tampilkan dataframe hasil deteksi
                if detections:
                    st.subheader(f"Deteksi Defect (Confidence ≥ {confidence_threshold})")
                    st.dataframe(detections)
                else:
                    st.info("Tidak ada defect yang terdeteksi.")

# Deteksi via Video
elif input_source == "Upload Video":
    st.header("Deteksi via Upload Video")
    uploaded_video = st.file_uploader("Pilih file video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Simpan video ke file sementara jika belum ada atau berbeda
        if 'video_path' not in st.session_state or st.session_state['video_name'] != uploaded_video.name:
            tfile = tempfile.NamedTemporaryFile(delete=False)
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
            st.error("Gagal membuka video.")
        else:
            fps = st.session_state['video_fps']
            frame_count = st.session_state['video_frame_count']

            st.write(f"FPS: {fps}, Total Frame: {frame_count}")
            frame_number = st.slider('Frame', 0, frame_count-1, 0)
            
            # Simpan frame number ke session state untuk mengecek perubahan
            if 'current_frame_number' not in st.session_state:
                st.session_state['current_frame_number'] = frame_number
                
            # Jika frame berubah atau threshold berubah, perbarui frame yang disimpan
            if frame_number != st.session_state['current_frame_number'] or 'current_frame' not in st.session_state:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if ret:
                    st.session_state['current_frame'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.session_state['current_frame_number'] = frame_number
            
            # Tampilkan frame yang disimpan dan hasil deteksi bersampingan
            if 'current_frame' in st.session_state:
                if model:
                    with st.spinner('Mendeteksi defect...'):
                        # Gunakan threshold saat ini
                        frame_with_boxes, detections = detect_defects(model, st.session_state['current_frame'], confidence_threshold)
                        
                        # Tampilkan frame asli dan hasil deteksi secara bersampingan
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"Frame Asli ({frame_number})")
                            st.image(st.session_state['current_frame'], use_container_width=True)
                        
                        with col2:
                            st.subheader("Hasil Deteksi")
                            st.image(frame_with_boxes, use_container_width=True)
                        
                        if detections:
                            st.subheader(f"Deteksi Defect (Confidence ≥ {confidence_threshold})")
                            st.dataframe(detections)
                        else:
                            st.info("Tidak ada defect yang terdeteksi.")

        cap.release()

# Bersihkan file sementara ketika aplikasi shutdown
def cleanup():
    if 'video_path' in st.session_state and os.path.exists(st.session_state['video_path']):
        os.unlink(st.session_state['video_path'])

# Tambahkan fungsi cleanup ke session state
if 'cleanup' not in st.session_state:
    import atexit
    atexit.register(cleanup)
    st.session_state['cleanup'] = True

# Footer
st.sidebar.markdown("---")
st.sidebar.write("© 2025 Deteksi Defect Pakaian")
