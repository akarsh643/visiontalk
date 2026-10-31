import os
import streamlit as st
from PIL import Image
from predict import (
    load_all_artifacts,
    generate_caption,
    generate_caption_beam_search,
)

st.set_page_config(page_title="VisionTalk", page_icon="📷", layout="centered")

st.title("📷 VisionTalk: AI Image Caption Generator")
st.markdown("Upload an image to generate descriptive captions using **DenseNet201 + LSTM**.")

@st.cache_resource
def get_artifacts():
    return load_all_artifacts()

feat_extractor, caption_model, word_to_idx, idx_to_word, max_len = get_artifacts()

# Sidebar controls for decoding
st.sidebar.header("Decoding Parameters")
decoding_method = st.sidebar.radio("Select Strategy:", ("Greedy Search", "Beam Search"))

beam_k = 3
if decoding_method == "Beam Search":
    beam_k = st.sidebar.slider("Beam Width (k):", min_value=2, max_value=7, value=3, step=1)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption", type="primary"):
        temp_path = "temp_uploaded_img.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Generating caption using {decoding_method}..."):
            if decoding_method == "Greedy Search":
                caption = generate_caption(
                    temp_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len
                )
            else:
                caption = generate_caption_beam_search(
                    temp_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len, beam_index=beam_k
                )

        if os.path.exists(temp_path):
            os.remove(temp_path)

        st.success("### Generated Caption:")
        st.write(f"**{caption.capitalize()}**" if caption else "_No caption generated._")