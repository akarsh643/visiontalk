import os
import io
import json
import numpy as np
import streamlit as st
from PIL import Image
from gtts import gTTS

from predict import (
    load_all_artifacts,
    generate_caption,
    generate_caption_beam_search
)

st.set_page_config(
    page_title="VisionTalk - AI Image Captioning",
    page_icon="👁️",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .caption-box { padding: 1.2rem; border-radius: 8px; background-color: #f0f4f8; border-left: 5px solid #1E88E5; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">👁️ VisionTalk: Multimodal Caption Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Assistive Image Description using DenseNet201 + LSTM Sequence Modeling</div>', unsafe_allow_html=True)

@st.cache_resource
def get_cached_artifacts():
    return load_all_artifacts()

with st.spinner("Loading models and vocabulary..."):
    feat_extractor, caption_model, word_to_idx, idx_to_word, max_len = get_cached_artifacts()

# Sidebar options
st.sidebar.header("Decoding Parameters")
decoding_method = st.sidebar.radio(
    "Choose Search Strategy:",
    ("Beam Search (Fluent)", "Greedy Search (Fast)")
)

beam_width = 3
if decoding_method == "Beam Search (Fluent)":
    beam_width = st.sidebar.slider("Beam Width (k):", min_value=1, max_value=7, value=3, step=1)

enable_tts = st.sidebar.checkbox("Enable Audio Narration (TTS)", value=True)

# Main UI layout
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("1. Input Image")
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        temp_img_path = "temp_uploaded_img.jpg"
        image.convert("RGB").save(temp_img_path)

with col2:
    st.subheader("2. Generated Description")
    if uploaded_file is not None:
        if st.button("Generate Caption", type="primary", use_container_width=True):
            with st.spinner("Analyzing image features and generating sequence..."):
                if decoding_method == "Greedy Search (Fast)":
                    caption = generate_caption(temp_img_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len)
                else:
                    caption = generate_caption_beam_search(temp_img_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len, beam_index=beam_width)
            
            st.markdown(f"""
                <div class="caption-box">
                    <h4 style="margin: 0; color: #0D47A1;">Predicted Description:</h4>
                    <p style="font-size: 1.25rem; font-weight: 500; margin: 0.5rem 0 0 0; color: #212121;">"{caption}"</p>
                </div>
            """, unsafe_allow_html=True)

            # Audio narration for accessibility
            if enable_tts:
                st.write("")
                st.markdown("**🔊 Audio Output (Assistive Tech):**")
                
                clean_text = caption.strip()
                if not clean_text:
                    st.warning("Caption is empty, cannot generate audio.")
                else:
                    audio_path = "output_caption.mp3"
                    try:
                        with st.spinner("Generating speech audio..."):
                            tts = gTTS(text=clean_text, lang='en', slow=False)
                            tts.save(audio_path)
                        
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        
                        st.audio(audio_bytes, format='audio/mp3')
                        st.success("Audio synthesized successfully!")
                    except Exception as e:
                        st.error(f"TTS Error: {e}")
                        st.info("Tip: gTTS requires an active internet connection. If WSL cannot resolve external DNS, test connection with: ping google.com")

            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
    else:
        st.info("Upload an image on the left to see caption generation and hear audio playback.")