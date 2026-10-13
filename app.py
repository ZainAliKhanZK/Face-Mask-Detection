import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
 
# ---------- Hugging Face model location (CHANGE THESE 2 LINES) ----------
HF_REPO_ID = "ZainAliKhanZAK/Face-Mask-Detector"   # e.g. "zainali/face-mask-detection"
HF_FILENAME = "mask_model.keras"              # exact file name on Hugging Face
 

# ---------- Page setup ----------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered",
)

# ---------- Custom styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 760px;}
    #MainMenu, footer {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        padding: 2rem 1.5rem;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800;}
    .hero p {margin: 0.4rem 0 0; opacity: 0.92; font-size: 1.05rem;}

    .result-card {
        padding: 1.4rem 1rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
        color: white;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
    }
    .result-card.ok {background: linear-gradient(135deg, #059669, #10b981);}
    .result-card.bad {background: linear-gradient(135deg, #dc2626, #f97316);}
    .result-card .emoji {font-size: 2.6rem;}
    .result-card .label {font-size: 1.8rem; font-weight: 800; margin: 0.2rem 0;}
    .result-card .conf {font-size: 1.05rem; opacity: 0.95;}

    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model ----------
@st.cache_resource
def get_model():
    return load_model("mask_model.keras")


model = get_model()


def predict_image(img):
    img = img.convert("RGB").resize((128, 128))
    img = np.expand_dims(np.array(img), axis=0)
    prediction = model.predict(img, verbose=0)[0][0]

    if prediction < 0.5:
        return "Masked", 1 - prediction
    return "No Mask", prediction


def show_result(img):
    with st.spinner("Analyzing image..."):
        label, conf = predict_image(img)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image(img, use_container_width=True)
    with col2:
        if label == "Masked":
            st.markdown(
                f"""
                <div class="result-card ok">
                    <div class="emoji">✅</div>
                    <div class="label">Mask Detected</div>
                    <div class="conf">Confidence: {conf:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card bad">
                    <div class="emoji">⚠️</div>
                    <div class="label">No Mask</div>
                    <div class="conf">Confidence: {conf:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.progress(float(conf))


# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>😷 Face Mask Detection</h1>
        <p>Upload a photo or use your camera to check if a mask is worn</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "A deep learning model (CNN) trained on a face mask dataset "
        "to classify images as **Masked** or **No Mask**."
    )
    st.subheader("Tips for best results")
    st.markdown(
        "- Use good lighting\n"
        "- Keep the face clearly visible\n"
        "- One face per image works best"
    )
    st.caption("Built with TensorFlow & Streamlit")

# ---------- Input tabs ----------
tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Use Camera"])

with tab_upload:
    file = st.file_uploader(
        "Drag and drop or browse a face image",
        type=["jpg", "jpeg", "png"],
    )
    if file:
        show_result(Image.open(file))

with tab_camera:
    st.caption("Allow camera access when your browser asks. Works on laptop webcam and phone camera.")
    photo = st.camera_input("Take a picture")
    if photo:
        show_result(Image.open(photo))
