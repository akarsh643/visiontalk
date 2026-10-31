import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

# Suppress standard TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.densenet import preprocess_input

def load_all_artifacts(
    model_path="models/best_model.keras",
    vocab_path="data/word_to_idx.json",
    meta_path="data/dataset_metadata.json"
):
    """Loads feature extractor, trained caption model, vocabulary, and sequence metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary mapping not found at: {vocab_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at: {meta_path}")

    # DenseNet201 feature extractor without classification head
    base_model = tf.keras.applications.DenseNet201(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )
    feat_extractor = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.output)

    # Caption merge model
    caption_model = load_model(model_path, compile=False)

    # Vocabulary mappings
    with open(vocab_path, "r") as f:
        word_to_idx = json.load(f)
    idx_to_word = {int(v): k for k, v in word_to_idx.items()}

  # Load sequence metadata with fallback keys
    with open(meta_path, "r") as f:
        meta = json.load(f)
        max_len = meta.get("max_len") or meta.get("max_length") or meta.get("max_seq_len") or 34

    return feat_extractor, caption_model, word_to_idx, idx_to_word, max_len

def extract_features(img_path, feat_extractor):
    """Extracts a 1920-dimensional feature vector for a given image path."""
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_arr = np.array(img, dtype=np.float32)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    features = feat_extractor.predict(img_arr, verbose=0)
    return features

def generate_caption(
    img_path,
    feat_extractor,
    caption_model,
    word_to_idx,
    idx_to_word,
    max_len,
    repetition_penalty=0.5
):
    """Generates a caption using Greedy Search with repetition penalty."""
    features = extract_features(img_path, feat_extractor)
    in_text = "startseq"

    for _ in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding="post")

        yhat = caption_model.predict([features, sequence], verbose=0)[0]

        # Repetition penalty downweights previously emitted words
        for word in in_text.split():
            if word in word_to_idx:
                token_id = word_to_idx[word]
                yhat[token_id] *= repetition_penalty

        yhat_idx = int(np.argmax(yhat))
        word = idx_to_word.get(yhat_idx, None)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    final_words = [w for w in in_text.split() if w not in ["startseq", "endseq"]]
    return " ".join(final_words)

def generate_caption_beam_search(
    img_path,
    feat_extractor,
    caption_model,
    word_to_idx,
    idx_to_word,
    max_len,
    beam_index=3,
    repetition_penalty=0.5
):
    """Generates a caption using Beam Search with length normalization and repetition penalty."""
    features = extract_features(img_path, feat_extractor)

    start_token = word_to_idx.get("startseq", 1)
    end_token = word_to_idx.get("endseq", 2)
    candidates = [[[start_token], 0.0]]

    for step in range(max_len):
        all_candidates = []
        for seq, score in candidates:
            if seq[-1] == end_token:
                all_candidates.append([seq, score])
                continue

            padded_seq = pad_sequences([seq], maxlen=max_len, padding="post")
            preds = caption_model.predict([features, padded_seq], verbose=0)[0]

            # Repetition penalty applied to active branch tokens
            for token_id in seq:
                if token_id in (start_token, end_token):
                    continue
                preds[token_id] *= repetition_penalty

            log_preds = np.log(np.maximum(preds, 1e-12))
            top_k_indices = np.argsort(log_preds)[-beam_index:]

            for idx in top_k_indices:
                candidate = [seq + [idx], score + log_preds[idx]]
                all_candidates.append(candidate)

        ordered = sorted(
            all_candidates,
            key=lambda x: x[1] / (len(x[0]) ** 0.7),
            reverse=True
        )
        candidates = ordered[:beam_index]

        if all(cand[0][-1] == end_token for cand in candidates):
            break

    best_seq = candidates[0][0]
    final_caption = [
        idx_to_word[idx]
        for idx in best_seq
        if idx in idx_to_word and idx_to_word[idx] not in ["startseq", "endseq"]
    ]

    return " ".join(final_caption)

def main():
    parser = argparse.ArgumentParser(description="VisionTalk CLI Image Caption Generator")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--beam", type=int, default=1, help="Beam width (1 for greedy, >1 for beam search)")
    parser.add_argument("--penalty", type=float, default=0.5, help="Repetition penalty factor (0.0 to 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    print("[INFO] Loading model and vocabulary...")
    feat_extractor, caption_model, word_to_idx, idx_to_word, max_len = load_all_artifacts()

    print(f"[INFO] Processing image: {args.image}")
    if args.beam <= 1:
        print("[INFO] Decoding Mode: Greedy Search")
        caption = generate_caption(
            args.image,
            feat_extractor,
            caption_model,
            word_to_idx,
            idx_to_word,
            max_len,
            repetition_penalty=args.penalty
        )
    else:
        print(f"[INFO] Decoding Mode: Beam Search (k={args.beam})")
        caption = generate_caption_beam_search(
            args.image,
            feat_extractor,
            caption_model,
            word_to_idx,
            idx_to_word,
            max_len,
            beam_index=args.beam,
            repetition_penalty=args.penalty
        )

    print("\n" + "=" * 40)
    print(f"Generated Caption: {caption}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()