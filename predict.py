import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_PATH = "models/best_model.keras"
VOCAB_PATH = "data/word_to_idx.json"
METADATA_PATH = "data/dataset_metadata.json"


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

    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    features = feat_extractor.predict(img_arr, verbose=0).reshape(1, -1)

    start_word = "start" if "start" in word_to_idx else "startseq"
    in_text = start_word

    for _ in range(max_len):
        seq = [
            word_to_idx.get(w, word_to_idx.get("<unk>", 1))
            for w in in_text.split()
        ]
        seq = pad_sequences([seq], maxlen=max_len, padding="post")

        preds = caption_model.predict(
            {"image_input": features, "caption_input": seq},
            verbose=0,
        )
        pred_idx = int(np.argmax(preds[0]))
        pred_word = idx_to_word.get(pred_idx, None)

        if pred_word is None or pred_word in ["end", "endseq", "<unk>", "<pad>"]:
            break

        in_text += " " + pred_word

    words = [w for w in in_text.split() if w not in ["start", "startseq"]]
    return " ".join(words)


def generate_caption_beam_search(image_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len, beam_index=3):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Cannot find image: {image_path}")

    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    features = feat_extractor.predict(img_arr, verbose=0).reshape(1, -1)

    # Use 'start' or 'startseq'
    start_token = word_to_idx.get("start", word_to_idx.get("startseq", 1))
    end_token = word_to_idx.get("end", word_to_idx.get("endseq", 2))

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
                {"image_input": features, "caption_input": padded_seq},
                verbose=0,
            )[0]

            top_k_indices = np.argsort(preds)[-beam_index:]

            for idx in top_k_indices:
                cand_seq = seq + [int(idx)]
                cand_score = score - np.log(preds[idx] + 1e-10)
                all_candidates.append((cand_seq, cand_score))

        if not all_candidates:
            break

        ordered = sorted(all_candidates, key=lambda tup: tup[1] / max(len(tup[0]), 1))
        sequences = ordered[:beam_index]

    all_final = completed_sequences if completed_sequences else sequences
    best_seq = sorted(all_final, key=lambda tup: tup[1] / max(len(tup[0]), 1))[0][0]

    final_caption = []
    for idx in best_seq:
        word = idx_to_word.get(idx, "")
        if word in ["start", "end", "startseq", "endseq", "<unk>", "<pad>"]:
            continue
        final_caption.append(word)

    return " ".join(final_caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Caption Generator")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--beam", type=int, default=3, help="Beam width (default: 3)")
    args = parser.parse_args()

    feat_ext, cap_model, w2i, i2w, max_len = load_all_artifacts()

    greedy_cap = generate_caption(args.image, feat_ext, cap_model, w2i, i2w, max_len)
    beam_cap = generate_caption_beam_search(args.image, feat_ext, cap_model, w2i, i2w, max_len, beam_index=args.beam)

    print("\n" + "=" * 60)
    print(f"Image:        {args.image}")
    print(f"Greedy:       {greedy_cap.capitalize()}")
    print(f"Beam (k={args.beam}):  {beam_cap.capitalize()}")
    print("=" * 60 + "\n")