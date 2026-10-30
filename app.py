import json
import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(page_title="VisionTalk - AI Image Captioner", page_icon="📷", layout="centered")
st.title("📷 VisionTalk: AI Image Captioner")
st.markdown("Upload any picture to generate an automated description using **DenseNet201 + LSTM**.")


@st.cache_resource
def load_captioning_pipeline():
    base = DenseNet201(weights="imagenet")
    feat_extractor = Model(inputs=base.inputs, outputs=base.layers[-2].output)
    caption_model = load_model("models/best_model.keras")

    with open("data/word_to_idx.json", "r", encoding="utf-8") as f:
        w2i = json.load(f)
    i2w = {int(v): k for k, v in w2i.items()}

    with open("data/dataset_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return feat_extractor, caption_model, w2i, i2w, meta["max_seq_len"]


with st.spinner("Loading neural network models..."):
    feature_extractor, caption_model, word_to_idx, idx_to_word, max_seq_len = load_captioning_pipeline()


def generate_caption(img_features):
    in_text = "startseq"
    features_expanded = np.expand_dims(img_features, axis=0)

    for _ in range(max_seq_len):
        sequence = [word_to_idx.get(w, word_to_idx.get("<unk>")) for w in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_seq_len, padding="post")

        preds = caption_model.predict(
            {"image_input": features_expanded, "caption_input": sequence},
            verbose=0
        )
        predicted_idx = int(np.argmax(preds[0]))
        predicted_word = idx_to_word.get(predicted_idx, None)

        if predicted_word is None or predicted_word in ["endseq", "<unk>", "<pad>"]:
            break

        in_text += " " + predicted_word

    return in_text.replace("startseq", "").strip()


uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption", type="primary"):
        with st.spinner("Analyzing image features and generating caption..."):
            img_resized = image.resize((224, 224))
            img_arr = img_to_array(img_resized)
            img_arr = np.expand_dims(img_arr, axis=0)
            img_arr = preprocess_input(img_arr)

            img_feats = feature_extractor.predict(img_arr, verbose=0).reshape(-1)
            result_caption = generate_caption(img_feats)

        st.subheader("Generated Description:")
        st.success(f"**\"{result_caption.capitalize()}\"**")