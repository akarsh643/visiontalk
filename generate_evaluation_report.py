import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from predict import (
    load_all_artifacts,
    generate_caption,
    generate_caption_beam_search
)

def load_test_data(captions_file="data/captions.txt", sample_size=100, seed=42):
    """Loads ground-truth captions and splits a subset for evaluation."""
    image_to_captions = {}
    with open(captions_file, "r") as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                img_name = parts[0]
                caption = ",".join(parts[1:]).strip()
                if img_name not in image_to_captions:
                    image_to_captions[img_name] = []
                image_to_captions[img_name].append(caption)

    # Filter to images that actually exist on disk
    available_images = [
        img for img in image_to_captions.keys()
        if os.path.exists(os.path.join("data/Images", img))
    ]

    random.seed(seed)
    test_images = random.sample(available_images, min(sample_size, len(available_images)))
    return test_images, image_to_captions

def compute_corpus_bleu(test_images, image_to_captions, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len):
    """Calculates smoothed corpus-level BLEU-1, BLEU-2, BLEU-3, BLEU-4."""
    references = []
    hypotheses_greedy = []
    hypotheses_beam = []

    print("\n[INFO] Evaluating test subset for BLEU benchmarks...")
    for img_name in tqdm(test_images):
        img_path = os.path.join("data/Images", img_name)
        
        # Ground truth tokenized references
        raw_refs = image_to_captions[img_name]
        tokenized_refs = [ref.lower().split() for ref in raw_refs]
        references.append(tokenized_refs)

        # Greedy hypothesis
        pred_greedy = generate_caption(img_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len)
        hypotheses_greedy.append(pred_greedy.lower().split())

        # Beam hypothesis (k=3)
        pred_beam = generate_caption_beam_search(img_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len, beam_index=3)
        hypotheses_beam.append(pred_beam.lower().split())

    smooth = SmoothingFunction().method1

    results = {
        "Greedy": {
            "BLEU-1": corpus_bleu(references, hypotheses_greedy, weights=(1.0, 0, 0, 0), smoothing_function=smooth),
            "BLEU-2": corpus_bleu(references, hypotheses_greedy, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
            "BLEU-3": corpus_bleu(references, hypotheses_greedy, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
            "BLEU-4": corpus_bleu(references, hypotheses_greedy, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
        },
        "Beam (k=3)": {
            "BLEU-1": corpus_bleu(references, hypotheses_beam, weights=(1.0, 0, 0, 0), smoothing_function=smooth),
            "BLEU-2": corpus_bleu(references, hypotheses_beam, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
            "BLEU-3": corpus_bleu(references, hypotheses_beam, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
            "BLEU-4": corpus_bleu(references, hypotheses_beam, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
        }
    }
    return results

def generate_10_image_grid(test_images, image_to_captions, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len, output_path="evaluation_report_grid.png"):
    """Generates and saves the 10-image visual comparison grid required by the rubric."""
    selected_10 = test_images[:10]
    fig, axes = plt.subplots(5, 2, figsize=(18, 25))
    axes = axes.flatten()

    print("\n[INFO] Generating 10-Image Visual Comparison Grid...")
    for idx, img_name in enumerate(selected_10):
        img_path = os.path.join("data/Images", img_name)
        img = Image.open(img_path)

        pred_greedy = generate_caption(img_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len)
        pred_beam = generate_caption_beam_search(img_path, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len, beam_index=3)
        actual_first_ref = image_to_captions[img_name][0]

        axes[idx].imshow(img)
        axes[idx].axis("off")
        
        caption_text = (
            f"Actual: {actual_first_ref}\n"
            f"Greedy: {pred_greedy}\n"
            f"Beam (k=3): {pred_beam}"
        )
        axes[idx].set_title(caption_text, fontsize=10, loc="left", wrap=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Visual report saved to {output_path}")

def main():
    feat_extractor, caption_model, word_to_idx, idx_to_word, max_len = load_all_artifacts()
    test_images, image_to_captions = load_test_data(sample_size=50)

    # 1. Compute & display BLEU benchmark
    metrics = compute_corpus_bleu(test_images, image_to_captions, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len)
    
    print("\n=================== BLEU EVALUATION BENCHMARK ===================")
    print(f"{'Method':<12} | {'BLEU-1':<8} | {'BLEU-2':<8} | {'BLEU-3':<8} | {'BLEU-4':<8}")
    print("-" * 55)
    for method, scores in metrics.items():
        print(f"{method:<12} | {scores['BLEU-1']:.4f}   | {scores['BLEU-2']:.4f}   | {scores['BLEU-3']:.4f}   | {scores['BLEU-4']:.4f}")
    print("=================================================================\n")

    # Save metrics to json for documentation
    with open("bleu_benchmark_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 2. Generate 10-image qualitative report
    generate_10_image_grid(test_images, image_to_captions, feat_extractor, caption_model, word_to_idx, idx_to_word, max_len)

if __name__ == "__main__":
    main()