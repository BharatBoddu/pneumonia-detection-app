import streamlit as st
import numpy as np
import cv2
import io
import tensorflow as tf
import requests, os
from tensorflow.keras.models import load_model

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

# Google Drive direct download link
MODEL_ID = "1aMbopMD9IocZAlfwgous7ml864xv7OfA"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
MODEL_PATH = "best_pneumonia_model.keras"

# Download if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load the model
model = load_model(MODEL_PATH)

#@st.cache_resource
#def load_model():
#    return tf.keras.models.load_model(MODEL_PATH)

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
    model = load_model()
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
