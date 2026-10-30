# VisionTalk: Image Captioning with DenseNet201 & LSTM

An end-to-end Deep Learning system built using TensorFlow/Keras on the Flickr8k dataset to generate natural language descriptions for input images.

## Architecture
* **Visual Encoder**: Pre-trained **DenseNet201** (ImageNet weights) stripped of the final classification layer (1920-dimensional output vectors).
* **Text Decoder**: Embedding layer combined with an **LSTM** layer for autoregressive sequence modeling.
* **Fusion Head**: Dense projection layers merging visual and textual representations with softmax probability outputs.

## Training Details
* **Dataset**: Flickr8k (8,000 images, 5 captions each)
* **Loss**: Categorical Crossentropy
* **Optimizer**: Adam
* **Best Model**: Epoch 13 checkpoint (`models/best_model.keras`)

## Evaluation (BLEU Scores)
Evaluated across the held-out test split using corpus-level n-gram precision:
* **BLEU-1**: ~0.54
* **BLEU-2**: ~0.34
* **BLEU-3**: ~0.22
* **BLEU-4**: ~0.16

## How to Run

### 1. Command-Line Interface
```bash
python predict.py --image "data/Images/sample.jpg"