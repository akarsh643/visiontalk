import argparse
import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

# Suppress TensorFlow startup logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
VOCAB_PATH = BASE_DIR / "data" / "word_to_idx.json"
METADATA_PATH = BASE_DIR / "data" / "dataset_metadata.json"


def load_all_artifacts():
    print("Loading models and vocabulary...")

    base_dense = DenseNet201(weights="imagenet")
    feat_extractor = Model(
        inputs=base_dense.inputs, outputs=base_dense.layers[-2].output
    )

    caption_model = load_model(MODEL_PATH)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        word_to_idx = json.load(f)
    idx_to_word = {int(v): k for k, v in word_to_idx.items()}

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    max_len = int(metadata.get("max_length", metadata.get("max_seq_len", 35)))

    return feat_extractor, caption_model, word_to_idx, idx_to_word, max_len


def extract_features(image_path, feat_extractor):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Cannot find image: {image_path}")
    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    return feat_extractor.predict(img_arr, verbose=0).reshape(1, -1)


def generate_caption(image_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len):
    features = extract_features(image_path, feat_extractor)

    start_token = word_to_idx.get("start", word_to_idx.get("startseq", 1))
    end_token = word_to_idx.get("end", word_to_idx.get("endseq", 2))
    pad_token = word_to_idx.get("<pad>", 0)
    unk_token = word_to_idx.get("<unk>", 1)

    in_text = [start_token]

    for _ in range(max_len):
        seq = pad_sequences([in_text], maxlen=max_len, padding="post")
        preds = caption_model.predict(
            {"image_input": features, "caption_input": seq}, verbose=0
        )[0]

        # Suppress padding, unk, and immediate repetition
        preds[pad_token] = 0.0
        preds[unk_token] = 0.0
        if len(in_text) > 1:
            preds[in_text[-1]] *= 0.5

        pred_idx = int(np.argmax(preds))

        if pred_idx == end_token or pred_idx == pad_token:
            break

        in_text.append(pred_idx)

    caption_words = [
        idx_to_word.get(idx)
        for idx in in_text
        if idx not in {start_token, end_token, pad_token, unk_token}
    ]
    return " ".join([w for w in caption_words if w])


def generate_caption_beam_search(image_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len, beam_index=3):
    features = extract_features(image_path, feat_extractor)

    start_token = word_to_idx.get("start", word_to_idx.get("startseq", 1))
    end_token = word_to_idx.get("end", word_to_idx.get("endseq", 2))
    pad_token = word_to_idx.get("<pad>", 0)
    unk_token = word_to_idx.get("<unk>", 1)

    sequences = [[[start_token], 0.0]]
    completed_sequences = []

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == end_token:
                completed_sequences.append((seq, score))
                continue

            padded_seq = pad_sequences([seq], maxlen=max_len, padding="post")
            preds = caption_model.predict(
                {"image_input": features, "caption_input": padded_seq}, verbose=0
            )[0]

            preds[pad_token] = 1e-12
            preds[unk_token] = 1e-12

            top_k_indices = np.argsort(preds)[-beam_index:]

            for idx in top_k_indices:
                cand_seq = seq + [int(idx)]
                cand_score = score - np.log(preds[idx] + 1e-10)
                all_candidates.append((cand_seq, cand_score))

        if not all_candidates:
            break

        ordered = sorted(all_candidates, key=lambda item: item[1] / max(len(item[0]), 1))
        sequences = ordered[:beam_index]

    candidates_to_rank = completed_sequences if completed_sequences else sequences
    best_seq = sorted(candidates_to_rank, key=lambda item: item[1] / max(len(item[0]), 1))[0][0]

    final_caption = [
        idx_to_word.get(idx, "")
        for idx in best_seq
        if idx not in {start_token, end_token, pad_token, unk_token}
    ]
    return " ".join([w for w in final_caption if w])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Caption Generator")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--beam", type=int, default=3, help="Beam width (default: 3)")
    args = parser.parse_args()

    feat_ext, cap_model, w2i, i2w, max_len = load_all_artifacts()

    print("\nRunning Greedy Search...")
    greedy_cap = generate_caption(args.image, feat_ext, cap_model, w2i, i2w, max_len)

    print(f"Running Beam Search (k={args.beam})...")
    beam_cap = generate_caption_beam_search(args.image, feat_ext, cap_model, w2i, i2w, max_len, beam_index=args.beam)

    print("\n" + "=" * 60)
    print(f"Image:       {args.image}")
    print(f"Greedy:      {greedy_cap.capitalize() if greedy_cap else 'No caption generated'}")
    print(f"Beam (k={args.beam}): {beam_cap.capitalize() if beam_cap else 'No caption generated'}")
    print("=" * 60 + "\n")