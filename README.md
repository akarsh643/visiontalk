# VisionTalk: AI Image Caption Generator

An end-to-end Deep Learning pipeline that automatically generates natural language descriptions for input images using a **CNN-LSTM (DenseNet201 + LSTM)** architecture trained on the Flickr8k dataset.

---

## Architecture Overview

- **Feature Extractor (Encoder)**: Pretrained `DenseNet201` (ImageNet weights) extracting 1920-dimensional feature representations.
- **Language Model (Decoder)**: Embedding layer + LSTM sequence processor with Dense projection.
- **Decoding Strategies**:
  - **Greedy Search**: Selects the highest probability token at each time step.
  - **Beam Search**: Evaluates top-$k$ sequence candidates using length-normalized log-probabilities.

---

## Project Structure

```text
VisionTalk/
├── data/
│   ├── Images/                # Flickr8k raw images
│   ├── captions.txt           # Raw dataset captions
│   ├── word_to_idx.json       # Vocabulary mapping
│   └── dataset_metadata.json  # Sequence lengths & hyperparameters
├── models/
│   └── best_model.keras       # Trained checkpoint
├── notebooks/
│   ├── 01_test_setup.ipynb
│   ├── 02_dataset_exploration.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_model_architecture.ipynb
│   ├── 05_data_generator.ipynb
│   ├── 06_model_training.ipynb
│   └── 07_inference_and_evaluation.ipynb
├── app.py                     # Streamlit Web Application
├── predict.py                 # Standalone CLI inference script
├── requirements.txt           # Project dependencies
└── README.md