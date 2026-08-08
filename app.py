"""
EcoPulse - Streamlit demo

"""
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# Set page configuration
st.set_page_config(page_title="EcoPulse", layout="centered")

st.title("🐘 EcoPulse: Elephant Detection")
st.write("Upload an image to detect elephants in the wild.")

@st.cache_resource
def get_yolo_model():
    """Loads the custom model if available, otherwise defaults to yolov8n.pt"""
    model_path = Path("best.pt")
    if model_path.exists():
        st.sidebar.success("Using custom model: best.pt")
        return YOLO("best.pt")
    else:
        st.sidebar.warning("Custom model not found. Using default YOLOv8n.")
        return YOLO("yolov8n.pt")

# Load model
model = get_yolo_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect"):
        with st.spinner("Analyzing..."):
            # Run inference
            results = model.predict(source=image, conf=0.25)
            
            # Plot the results
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detection Results", use_container_width=True)
            
            # Show detection count
            count = len(results[0].boxes)
            st.success(f"Detected {count} elephant(s) in the image.")
