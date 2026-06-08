# =======================
# IMPORTS
# =======================
import hashlib
import io
import os

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageFile

from auth import (
    authenticate_user,
    create_login_session,
    create_user,
    delete_login_session,
    delete_user,
    get_password_validation_error,
    get_user_by_session_token,
    is_valid_email,
    is_valid_name,
    update_user_name,
    update_user_password,
)
from db import get_scan_history, init_db, save_scan_record
from gradcam_utils import GradCAM
from model_utils import CLASS_NAMES, get_transform, load_model, predict_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(page_title="Brain Tumor MRI Classification", layout="wide")

# =======================
# PROFESSIONAL UI
# =======================
st.markdown(
    """
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

.auth-card {
    max-width: 520px;
    margin: 0 auto 18px auto;
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
""",
    unsafe_allow_html=True,
)

# =======================
# HEADER
# =======================
st.markdown(
    """
<div class="header">
    <h1>Brain Tumor MRI Classification</h1>
    <p>Create an account, log in, upload a brain MRI scan, and keep a saved history of analysis results.</p>
</div>
""",
    unsafe_allow_html=True,
)

CLASS_COLORS = {
    "glioma": "#ff4d4f",
    "meningioma": "#f59e0b",
    "notumor": "#22c55e",
    "pituitary": "#3b82f6",
}

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}


# =======================
# DATABASE
# =======================
try:
    init_db()
except Exception as e:
    st.error("Database connection failed. Please start MySQL and check your database settings.")
    st.info(
        "Default settings: host=localhost, port=3306, user=root, password empty, database=brain_tumor_app. "
        "You can override these with environment variables DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME "
        "or Streamlit secrets."
    )
    st.exception(e)
    st.stop()


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
        return load_model(model_path, device)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()


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


def login_user(user):
    st.session_state["logged_in"] = True
    st.session_state["user_id"] = user["id"]
    st.session_state["user_name"] = user["name"]
    st.session_state["user_email"] = user["email"]


def get_session_token_from_url():
    token = st.query_params.get("session")
    if isinstance(token, list):
        return token[0] if token else None
    return token


def persist_login(user):
    token = create_login_session(user["id"])
    st.session_state["session_token"] = token
    st.query_params["session"] = token


def restore_login_from_url():
    if st.session_state.get("logged_in"):
        return

    token = get_session_token_from_url()
    user = get_user_by_session_token(token)
    if user:
        login_user(user)
        st.session_state["session_token"] = token
    elif token:
        st.query_params.clear()


def logout_user():
    token = st.session_state.get("session_token") or get_session_token_from_url()
    delete_login_session(token)
    st.query_params.clear()

    for key in ["logged_in", "user_id", "user_name", "user_email", "last_saved_scan", "session_token"]:
        st.session_state.pop(key, None)


def show_auth_page():
    st.markdown('<div class="card auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Account Access</div>', unsafe_allow_html=True)

    if st.session_state.pop("account_created", False):
        st.success("Account created successfully. Please log in with your email and password.")

    login_tab, signup_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")

        if submitted:
            if not email or not password:
                st.warning("Please enter email and password.")
            elif not is_valid_email(email):
                st.error("Please enter a valid email address.")
            else:
                user = authenticate_user(email, password)
                if user:
                    login_user(user)
                    persist_login(user)
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with signup_tab:
        with st.form("signup_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account")

        if submitted:
            password_error = get_password_validation_error(password)

            if not name or not email or not password:
                st.warning("Please fill all required fields.")
            elif not is_valid_name(name):
                st.error("Full name must be 3-80 letters and cannot contain numbers or invalid symbols.")
            elif not is_valid_email(email):
                st.error("Please enter a valid email address.")
            elif not confirm_password:
                st.error("Please confirm your password.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif password_error:
                st.error(password_error)
            else:
                user = create_user(name, email, password)
                if user:
                    st.session_state["account_created"] = True
                    st.rerun()
                else:
                    st.error("This email is already registered.")

    st.markdown("</div>", unsafe_allow_html=True)


def show_account_tools():
    st.sidebar.title("Account")
    st.sidebar.success(f"Logged in as {st.session_state['user_name']}")
    st.sidebar.caption(st.session_state["user_email"])

    with st.sidebar.expander("Account Settings"):
        with st.form("update_name_form"):
            new_name = st.text_input("Name", value=st.session_state["user_name"])
            update_name = st.form_submit_button("Update Name")

        if update_name:
            if not is_valid_name(new_name):
                st.error("Full name must be 3-80 letters and cannot contain numbers or invalid symbols.")
            elif update_user_name(st.session_state["user_id"], new_name):
                st.session_state["user_name"] = new_name.strip()
                st.success("Name updated.")
                st.rerun()
            else:
                st.error("Name could not be updated.")

        with st.form("password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            change_password = st.form_submit_button("Change Password")

        if change_password:
            password_error = get_password_validation_error(new_password)
            if password_error:
                st.error(password_error)
            elif update_user_password(st.session_state["user_id"], current_password, new_password):
                st.success("Password updated.")
            else:
                st.error("Current password is incorrect.")

        delete_confirm = st.checkbox("I understand this will delete my account and scan history.")
        if st.button("Delete Account", disabled=not delete_confirm):
            if delete_user(st.session_state["user_id"]):
                logout_user()
                st.success("Account deleted.")
                st.rerun()
            else:
                st.error("Account could not be deleted.")

    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()


def show_scan_history():
    st.markdown("## Scan History")
    rows = get_scan_history(st.session_state["user_id"])

    if not rows:
        st.info("No scan records yet.")
        return

    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "File": row["file_name"],
                "Prediction": CLASS_LABELS.get(row["predicted_class"], row["predicted_class"]),
                "Confidence": f"{float(row['confidence']) * 100:.1f}%",
                "Date": row["created_at"].strftime("%Y-%m-%d %H:%M"),
            }
        )
    st.table(display_rows)


def save_prediction_once(image_bytes, file_name, predicted_class, confidence):
    file_hash = hashlib.sha256(image_bytes).hexdigest()
    signature = f"{st.session_state['user_id']}:{file_hash}:{predicted_class}:{confidence:.6f}"

    if st.session_state.get("last_saved_scan") == signature:
        return

    save_scan_record(st.session_state["user_id"], file_name, predicted_class, confidence)
    st.session_state["last_saved_scan"] = signature


def show_dashboard():
    model = load()
    show_account_tools()

    st.sidebar.title("Scan Analysis")
    uploaded = st.sidebar.file_uploader(
        "Upload brain MRI scan",
        type=["jpg", "jpeg", "png", "jfif", "bmp", "tiff", "webp"],
    )

    confidence_threshold = st.sidebar.slider(
        "Minimum Confidence",
        0.0,
        1.0,
        0.6,
        help="Predictions below this confidence will be marked as uncertain.",
    )

    st.sidebar.markdown("---")
    st.sidebar.info("For best results, upload a clear axial, coronal, or sagittal brain MRI image.")

    st.markdown(f"### Welcome, {st.session_state['user_name']}")

    if not uploaded:
        st.info("Upload a brain MRI scan from the sidebar to start analysis.")
        show_scan_history()
        return

    col1, col2 = st.columns([1, 1])

    image_bytes = uploaded.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Uploaded MRI Scan</div>', unsafe_allow_html=True)
        st.image(image, caption="Source image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    pred, probs = predict_image(model, image, device, confidence_threshold)

    if pred == "invalid_image":
        show_response_popup(
            "Image Not Supported",
            "This picture does not look like a valid brain MRI scan. The app can detect brain tumor classes only from brain MRI images.",
            "error",
        )
        st.stop()

    if pred == "low_confidence":
        top_idx = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_confidence = float(probs[top_idx]) * 100
        show_response_popup(
            "Uncertain Result",
            f"The model confidence is below the selected threshold. Best estimate: {top_class.upper()} ({top_confidence:.1f}%). Please upload a clearer MRI scan for review.",
            "warning",
        )
        st.stop()

    predicted_class = CLASS_NAMES[pred]
    confidence = float(probs[pred])
    save_prediction_once(image_bytes, uploaded.name, predicted_class, confidence)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analysis Summary</div>', unsafe_allow_html=True)

        if predicted_class == "notumor":
            st.markdown('<div class="result-badge safe">No tumor pattern detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-badge alert">Tumor pattern detected</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
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
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Class Probability Breakdown</div>', unsafe_allow_html=True)

        for i, cls in enumerate(CLASS_NAMES):
            value = float(probs[i])
            color = CLASS_COLORS[cls]

            st.markdown(f"**{CLASS_LABELS[cls]}**")
            st.markdown(
                f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width:{value * 100}%; background:{color};">
                    {value * 100:.1f}%
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

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
            cam = cv2.resize(cam, (224, 224))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(image.resize((224, 224))), 0.6, heatmap, 0.4, 0)

            col_a, col_b = st.columns(2)

            with col_a:
                st.image(image, caption="Original scan")

            with col_b:
                st.image(overlay, caption="Grad-CAM heatmap")

        except Exception:
            st.error("Grad-CAM could not be generated for this scan. Please try another image.")

    show_scan_history()


restore_login_from_url()

if st.session_state.get("logged_in"):
    show_dashboard()
else:
    show_auth_page()

# =======================
# FOOTER
# =======================
st.markdown(
    """
<hr>
<p style='text-align:center; color:#64748b;'>Brain Tumor MRI Classification | Deep Learning Decision Support | 2026</p>
""",
    unsafe_allow_html=True,
)
