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
st.set_page_config(page_title="Brain Tumor MRI Classification", layout="wide")

# =======================
# PROFESSIONAL UI
# =======================
st.markdown("""
<style>

.stApp {
    background: #f6f8fb;
}

html, body, [class*="css"] {
    color: #172033;
    font-family: "Inter", "Segoe UI", sans-serif;
}

.header {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 10px 28px rgba(31, 41, 55, 0.06);
}

.header h1 {
    color: #0f172a;
    font-size: 2rem;
    line-height: 1.2;
    margin: 0 0 8px 0;
    letter-spacing: 0;
}

.header p {
    color: #526071;
    font-size: 1rem;
    margin: 0;
}

.card {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 22px rgba(31, 41, 55, 0.05);
}

.section-title {
    color: #0f172a;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 14px 0;
}

.result-badge {
    border-radius: 8px;
    padding: 12px 14px;
    font-weight: 700;
    margin-bottom: 16px;
}

.result-badge.safe {
    color: #166534;
    background: #dcfce7;
    border: 1px solid #86efac;
}

.result-badge.alert {
    color: #991b1b;
    background: #fee2e2;
    border: 1px solid #fecaca;
}

.metric-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 18px;
}

.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
}

.metric-label {
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.metric-value {
    color: #0f172a;
    font-size: 1.25rem;
    font-weight: 800;
}

section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #dbe3ef;
}

.stButton>button {
    background: #2563eb;
    color: white !important;
    border-radius: 8px;
    border: 0;
    font-weight: 700;
    padding: 0.55rem 1rem;
}

.progress-bar {
    background: #e5e7eb;
    border-radius: 999px;
    height: 20px;
    overflow: hidden;
    margin: 6px 0 12px 0;
}

.progress-fill {
    height: 100%;
    color: #ffffff;
    font-size: 0.78rem;
    font-weight: 700;
    line-height: 20px;
    min-width: 44px;
    padding-right: 8px;
    text-align: right;
}

</style>
""", unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.markdown("""
<div class="header">
    <h1>Brain Tumor MRI Classification</h1>
    <p>Upload a brain MRI scan to classify the image and review model confidence with visual explainability.</p>
</div>
""", unsafe_allow_html=True)

# =======================
# MODEL
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="Loading AI model...")
def load():
    model_path = "final_model_11111.pth"

    if not os.path.exists(model_path):
        st.error("Model file was not found. Please confirm final_model_11111.pth is available in the project folder.")
        st.stop()

    try:
        model = load_model(model_path, device)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

# Load model safely
model = load()


def show_response_popup(title, message, level="warning"):
    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _popup():
            if level == "error":
                st.error(message)
            elif level == "success":
                st.success(message)
            else:
                st.warning(message)

        _popup()
    else:
        st.toast(message)
# =======================
# SIDEBAR
# =======================
st.sidebar.title("Scan Analysis")

uploaded = st.sidebar.file_uploader(
    "Upload brain MRI scan",
    type=["jpg","jpeg","png","jfif","bmp","tiff","webp"]
)

confidence_threshold = st.sidebar.slider(
    "Minimum Confidence",
    0.0,
    1.0,
    0.6,
    help="Predictions below this confidence will be marked as uncertain."
)

st.sidebar.markdown("---")
st.sidebar.info("For best results, upload a clear axial, coronal, or sagittal brain MRI image.")

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
        st.markdown('<div class="section-title">Uploaded MRI Scan</div>', unsafe_allow_html=True)
        st.image(image, caption="Source image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # PREDICTION
    pred, probs = predict_image(model, image, device, confidence_threshold)

    if pred == "invalid_image":
        show_response_popup(
            "Image Not Supported",
            "The uploaded file does not appear to be a valid brain MRI scan. Please upload a clear MRI image and try again.",
            "error"
        )
        st.stop()

    if pred == "low_confidence":
        top_idx = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_confidence = float(probs[top_idx]) * 100
        show_response_popup(
            "Uncertain Result",
            f"The model confidence is below the selected threshold. Best estimate: {top_class.upper()} ({top_confidence:.1f}%). Please upload a clearer MRI scan for review.",
            "warning"
        )
        st.stop()


    predicted_class = CLASS_NAMES[pred]
    confidence = float(probs[pred])

    CLASS_COLORS = {
        "glioma": "#ff4d4f",
        "meningioma": "#f59e0b",
        "notumor": "#22c55e",
        "pituitary": "#3b82f6"
    }

    CLASS_LABELS = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "No Tumor",
        "pituitary": "Pituitary"
    }

    # RESULTS
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analysis Summary</div>', unsafe_allow_html=True)

        if predicted_class == "notumor":
            st.markdown('<div class="result-badge safe">No tumor pattern detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-badge alert">Tumor pattern detected</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">Predicted Class</div>
                <div class="metric-value">{CLASS_LABELS[predicted_class]}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence * 100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Class Probability Breakdown</div>', unsafe_allow_html=True)

        for i, cls in enumerate(CLASS_NAMES):
            value = float(probs[i])
            color = CLASS_COLORS[cls]

            st.markdown(f"**{CLASS_LABELS[cls]}**")
            st.markdown(f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width:{value*100}%; background:{color};">
                    {value * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =======================
    # GRADCAM
    # =======================
    st.markdown("## Model Explainability")

    if predicted_class == "notumor":
        st.info(
            "Grad-CAM is available only when a tumor class is detected. "
            "For a no-tumor result, the confidence score reflects the model's probability for the No Tumor class."
        )
    elif st.button("Generate Grad-CAM Heatmap"):
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
                st.image(image, caption="Original scan")

            with colB:
                st.image(overlay, caption="Grad-CAM heatmap")

        except:
            st.error("Grad-CAM could not be generated for this scan. Please try another image.")

# =======================
# FOOTER
# =======================
st.markdown("""
<hr>
<p style='text-align:center; color:#64748b;'>Brain Tumor MRI Classification | Deep Learning Decision Support | 2026</p>
""", unsafe_allow_html=True)
