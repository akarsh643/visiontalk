import os
import tempfile
import requests
import streamlit as st
from PIL import Image
from gtts import gTTS

# Configure application shell
st.set_page_config(
    page_title="VisionTalk: Assistive Captioning",
    page_icon="👁️",
    layout="wide"
)

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8080")

st.title("👁️ VisionTalk: AI Image Captioning")
st.markdown("Accessible, AI-powered image descriptions serving assistive narration.")

# Sidebar Configuration
st.sidebar.header("Inference Settings")

decoding_strategy = st.sidebar.selectbox(
    "Decoding Strategy",
    options=["Greedy Search (Fast)", "Beam Search (Fluent)"],
    index=1
)

beam_width = 3
if decoding_strategy == "Beam Search (Fluent)":
    beam_width = st.sidebar.slider("Beam Width (k)", min_value=2, max_value=7, value=3, step=1)
else:
    beam_width = 1

repetition_penalty = st.sidebar.slider(
    "Repetition Penalty",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Values below 1.0 penalize repeated tokens to avoid loops."
)

# Check API Health Status
try:
    health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
    api_online = health_resp.status_code == 200 and health_resp.json().get("model_loaded", False)
except requests.exceptions.RequestException:
    api_online = False

if api_online:
    st.sidebar.success("Backend API: Online")
else:
    st.sidebar.error("Backend API: Offline. Ensure 'python api.py' is running on port 8080.")

# Main Interface Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a JPEG or PNG image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("2. Generated Description")

    if uploaded_file is not None:
        generate_button = st.button("Generate Caption & Audio", type="primary", disabled=not api_online)

        if generate_button:
            with st.spinner("Communicating with FastAPI inference server..."):
                try:
                    # Prepare file payload for multipart/form-data
                    uploaded_file.seek(0)
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    params = {
                        "beam": beam_width,
                        "penalty": repetition_penalty
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/predict",
                        files=files,
                        params=params,
                        timeout=30
                    )

                    if response.status_code == 200:
                        data = response.json()
                        caption = data.get("caption", "No caption returned.")
                        strategy = data.get("strategy", decoding_strategy)

                        st.success(f"Strategy: {strategy}")
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f2f6; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                                <h3 style="margin: 0; color: #1f2937;">{caption}</h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Audio narration synthesis via gTTS
                        with st.spinner("Synthesizing audio narration..."):
                            tts = gTTS(text=caption, lang="en", slow=False)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                tts.save(fp.name)
                                audio_path = fp.name

                            with open(audio_path, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format="audio/mp3")

                            if os.path.exists(audio_path):
                                os.remove(audio_path)

                    else:
                        st.error(f"Inference error ({response.status_code}): {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to communicate with API server: {e}")
    else:
        st.info("Upload an image on the left column to get started.")