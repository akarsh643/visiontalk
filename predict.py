import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hide unnecessary TensorFlow startup warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_PATH = "models/best_model.keras"
VOCAB_PATH = "data/word_to_idx.json"
METADATA_PATH = "data/dataset_metadata.json"


def load_all_artifacts():
    print("Loading models and vocabulary...")
    # Load DenseNet feature extractor
    base_dense = DenseNet201(weights="imagenet")
    feat_extractor = Model(
        inputs=base_dense.inputs, outputs=base_dense.layers[-2].output
    )

    # Load trained decoder model
    caption_model = load_model(MODEL_PATH)

    # Load tokenizer mappings
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        word_to_idx = json.load(f)
    idx_to_word = {int(v): k for k, v in word_to_idx.items()}

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return (
        feat_extractor,
        caption_model,
        word_to_idx,
        idx_to_word,
        metadata["max_seq_len"],
    )


def generate_caption(image_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Cannot find image: {image_path}")

    # 1. Resize and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    # 2. Extract visual features
    features = feat_extractor.predict(img_arr, verbose=0).reshape(1, -1)

    # 3. Predict tokens word by word
    in_text = "startseq"
    for _ in range(max_len):
        seq = [
            word_to_idx.get(w, word_to_idx.get("<unk>"))
            for w in in_text.split()
        ]
        seq = pad_sequences([seq], maxlen=max_len, padding="post")

        preds = caption_model.predict(
            {"image_input": features, "caption_input": seq},
            verbose=0,
        )
        pred_idx = int(np.argmax(preds[0]))
        pred_word = idx_to_word.get(pred_idx, None)

        # Stop if model reaches end or encounters unk/pad
        if pred_word is None or pred_word in ["endseq", "<unk>", "<pad>"]:
            break

        in_text += " " + pred_word

    return in_text.replace("startseq", "").strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Caption Generator")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()

    feat_ext, cap_model, w2i, i2w, max_len = load_all_artifacts()
    caption = generate_caption(args.image, feat_ext, cap_model, w2i, i2w, max_len)

    print("\n" + "=" * 50)
    print(f"Image:   {args.image}")
    print(f"Caption: {caption.capitalize()}")
    print("=" * 50 + "\n")