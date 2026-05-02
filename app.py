# =======================
# IMPORTS
# =======================
import os
import io
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image, ImageFile

from model_utils import load_model, predict_image, CLASS_NAMES, get_transform
from gradcam_utils import GradCAM

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(page_title="Brain Tumor Dashboard", page_icon="🧠", layout="wide")

# =======================
# GLASSMORPHISM UI
# =======================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: url('https://images.unsplash.com/photo-1530497610245-94d3c16cda28');
    background-size: cover;
    background-attachment: fixed;
}

/* BLUR OVERLAY */
.main::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(5,10,25,0.75);
    backdrop-filter: blur(12px);
    z-index: -1;
}

/* TEXT FIX */
html, body, [class*="css"] {
    color: #ffffff !important;
}

/* HEADER */
.header {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* CARD */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(25px);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.7) !important;
    backdrop-filter: blur(20px);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white !important;
    border-radius: 12px;
    font-weight: bold;
}

/* PROGRESS */
.progress-bar {
    background: rgba(30,41,59,0.7);
    border-radius: 10px;
    height: 18px;
}

</style>
""", unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.markdown("""
<div class="header">
    <h1>🧠 Brain Tumor Detection Dashboard</h1>
    <p>AI-powered MRI Analysis System</p>
</div>
""", unsafe_allow_html=True)

# =======================
# MODEL
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load():
    model_path = "final_model_11111.pth"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Check GitHub repo.")
        st.stop()

    return load_model(model_path, device)
# =======================
# SIDEBAR
# =======================
st.sidebar.title("⚙️ Controls")

uploaded = st.sidebar.file_uploader(
    "Upload MRI Image",
    type=["jpg","jpeg","png","jfif","bmp","tiff","webp"]
)

threshold = st.sidebar.slider("Glioma Sensitivity", 0.0, 1.0, 0.4)

st.sidebar.markdown("---")
st.sidebar.info("Upload MRI scan to detect tumor type")

# =======================
# MAIN
# =======================
if uploaded:

    col1, col2 = st.columns([1,1])

    image_bytes = uploaded.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # IMAGE
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption="MRI Scan", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # PREDICTION
    pred, probs = predict_image(model, image, device, threshold)

    if pred == "invalid_image":
        st.error("⚠️ Not a valid MRI image")
        st.stop()

    if pred == "low_confidence":
        st.warning("⚠️ Model uncertain. Try clearer MRI")
        st.stop()

    predicted_class = CLASS_NAMES[pred]

    CLASS_COLORS = {
        "glioma": "#ff4d4f",
        "meningioma": "#f59e0b",
        "notumor": "#22c55e",
        "pituitary": "#3b82f6"
    }

    # RESULTS
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if predicted_class == "notumor":
            st.success("✅ NO TUMOR DETECTED")
        else:
            st.error("🚨 TUMOR DETECTED")

        st.markdown(f"### 🧾 Type: **{predicted_class.upper()}**")
        st.markdown(f"### 🎯 Confidence: **{np.max(probs):.3f}**")

        st.markdown("### 📊 Class Probabilities")

        for i, cls in enumerate(CLASS_NAMES):
            value = float(probs[i])
            color = CLASS_COLORS[cls]

            st.markdown(f"**{cls}**")
            st.markdown(f"""
            <div class="progress-bar">
                <div style="width:{value*100}%; height:100%; background:{color}; text-align:right; padding-right:5px;">
                    {value:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =======================
    # GRADCAM
    # =======================
    st.markdown("## 🔥 Explainability (Grad-CAM)")

    if st.button("Generate Heatmap"):
        transform = get_transform()
        img_tensor = transform(image).unsqueeze(0).to(device)

        try:
            cam = GradCAM(model).generate(img_tensor, pred)
            cam = cv2.resize(cam, (224,224))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(
                np.array(image.resize((224,224))), 0.6, heatmap, 0.4, 0
            )

            colA, colB = st.columns(2)

            with colA:
                st.image(image, caption="Original")

            with colB:
                st.image(overlay, caption="GradCAM")

        except:
            st.error("GradCAM failed. Check model layer.")

# =======================
# FOOTER
# =======================
st.markdown("""
<hr>
<p style='text-align:center;'>🚀 Brain Tumor Detection | Deep Learning | 2026</p>
""", unsafe_allow_html=True)