import streamlit as st
import numpy as np
import cv2
import io
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from pathlib import Path
import requests

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_OK = True
except ImportError:
    PYDICOM_OK = False

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")
st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to check for pneumonia using AI.")
st.markdown("---")

# Hugging Face Model Hub (recommended for large models)
# Replace with your own Hugging Face repo: https://huggingface.co/<username>/<repo-name>
HF_MODEL_ID = "bharatboddu/pneumonia-detection-model"  # Update this!
HF_MODEL_FILE = "best_pneumonia_model.keras"

# Use Streamlit's cache directory for persistent storage
# On Streamlit Cloud, this persists across app reruns
CACHE_DIR = Path.home() / ".cache" / "streamlit_pneumonia"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = CACHE_DIR / HF_MODEL_FILE

@st.cache_resource
def load_pneumonia_model():
    try:
        if not MODEL_PATH.exists():
            st.info("📥 Downloading model from Hugging Face... This may take a moment.")
            try:
                # Download from Hugging Face Hub with streaming for large files
                hf_url = f"https://huggingface.co/{HF_MODEL_ID}/resolve/main/{HF_MODEL_FILE}"
                response = requests.get(hf_url, timeout=600, stream=True)  # 10 min timeout
                response.raise_for_status()
                
                # Get total file size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = min(downloaded / total_size, 1.0)
                                st.progress(progress, text=f"Downloaded {downloaded/(1024*1024):.1f}MB")
                    
            except Exception as e:
                st.error(f"❌ Download failed: {str(e)}")
                st.error("💡 Make sure your model is uploaded to Hugging Face Hub:")
                st.code(f"HF_MODEL_ID = '{HF_MODEL_ID}'")
                return None
            
            # Verify download
            if not MODEL_PATH.exists():
                st.error("Model file download failed. File not found.")
                return None
            
            file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            if file_size_mb < 1:
                st.error("Model file is empty. Download may have failed.")
                MODEL_PATH.unlink()
                return None
            
            st.success(f"✅ Model downloaded successfully! ({file_size_mb:.1f} MB)")
        else:
            file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            st.success(f"✅ Model loaded from cache! ({file_size_mb:.1f} MB)")
        
        return load_model(str(MODEL_PATH))
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_pneumonia_model()

if model is None:
    st.error("⚠️ Failed to load the model. Please refresh the page and try again.")
    st.stop()

def load_dicom(file_bytes, size=(128,128)):
    dcm = pydicom.dcmread(io.BytesIO(file_bytes))
    try:    px = apply_voi_lut(dcm.pixel_array, dcm)
    except: px = dcm.pixel_array.astype(float)
    px = px.astype(float)
    if getattr(dcm, "PhotometricInterpretation","") == "MONOCHROME1":
        px = px.max() - px
    px = (px - px.min()) / (px.max() - px.min() + 1e-8) * 255.0
    return cv2.resize(px.astype(np.uint8), size)

def preprocess(img, channels, size=(128,128)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, size).astype(np.float32) / 255.0
    if channels == 3:
        return np.repeat(resized[...,np.newaxis], 3, axis=-1)[np.newaxis,...]
    return resized[np.newaxis,...,np.newaxis]

# Sidebar
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.10, 0.90, 0.50, 0.05)
st.sidebar.markdown("---")
st.sidebar.warning("For educational use only. Not for clinical diagnosis.")

# File upload
uploaded_file = st.file_uploader("Choose a chest X-ray image",
                                   type=["dcm","png","jpg","jpeg"])

if uploaded_file is not None:
    if model is None:
        st.error("Model is not available. Please refresh the page.")
    else:
        file_bytes = uploaded_file.read()
        ext = uploaded_file.name.split(".")[-1].lower()

        if ext == "dcm":
            img_gray = load_dicom(file_bytes)
        else:
            from PIL import Image
            img_gray = np.array(Image.open(io.BytesIO(file_bytes)).convert("L"))

        in_channels = model.input_shape[-1]
        inp = preprocess(img_gray, in_channels)
        prob = float(model.predict(inp, verbose=0)[0][0])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded X-Ray")
            st.image(img_gray, caption=uploaded_file.name, use_column_width=True, clamp=True)

        with col2:
            st.subheader("Prediction Result")
            if prob >= threshold:
                st.error(f"**Pneumonia Detected**")
            else:
                st.success(f"**No Pneumonia**")

            st.metric("Pneumonia Probability", f"{prob*100:.1f}%")
            st.metric("Threshold", f"{threshold*100:.0f}%")
            st.progress(prob)

        st.markdown("---")
        st.markdown("**Class Probabilities:**")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["No Pneumonia","Pneumonia"], [1-prob, prob],
                 color=["green","red"], edgecolor="black", height=0.5)
        ax.axvline(threshold, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
else:
    st.info("Please upload a chest X-ray image (DICOM, PNG, or JPG) to get started.")
    st.markdown("""
**How to use:**
1. Upload a chest X-ray image
2. Adjust threshold in the sidebar if needed
3. View the prediction result and probability
    """)

st.markdown("---")
st.caption("Pneumonia Detection AI — Built with TensorFlow and Streamlit")
