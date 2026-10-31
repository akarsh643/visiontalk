# VisionTalk: End-to-End Image Captioning Engine

VisionTalk is an end-to-end deep learning system that takes an input image, generates descriptive natural language captions, and speaks the generated captions using text-to-speech narration.

The application uses a decoupled microservice architecture: a **FastAPI** backend running CNN + LSTM inference, paired with an interactive **Streamlit** frontend, orchestrated with **Docker Compose** and tested through **GitHub Actions**.

---

## Architecture Overview

```text
[ User / Browser ]
        │
        ▼ (Port 8501)
┌──────────────────────────────────────────────┐
│  Streamlit Frontend (visiontalk_ui)          │
│  - Image Upload & UI Controls                │
│  - Parameter Selection (Beam Width, Penalty) │
│  - Text-to-Speech Narration (gTTS)           │
└──────────────────────┬───────────────────────┘
                       │ HTTP / REST
                       ▼ (Port 8080)
┌──────────────────────────────────────────────┐
│  FastAPI Backend (visiontalk_api)            │
│  - DenseNet201 (Transfer Learning Encoder)   │
│  - Word Embedding + LSTM Decoder             │
│  - Beam Search with Repetition Penalty       │
└──────────────────────────────────────────────┘
```

---

## Key Features

- **Transfer Learning Feature Extractor**: Pre-trained DenseNet201 CNN extracting 1,920-dimensional image feature vectors.
- **Sequence Generation Decoder**: Recurrent Neural Network (LSTM) with learned word embeddings to predict caption tokens step-by-step.
- **Beam Search Decoding**: Configurable search width and repetition penalty to prevent generic or looping sentences.
- **Microservice Containerization**: Independent Docker containers for frontend and backend communicating over a custom bridge network.
- **Automated CI/CD**: Pytest test suite covering endpoints, lifespan loading, and invalid file detection, verified on every push via GitHub Actions.

---

## Project Structure

```text
VisionTalk/
├── .github/workflows/
│   └── ci.yml                   # GitHub Actions automated test & build pipeline
├── data/
│   ├── dataset_metadata.json    # Vocabulary size, max caption length, special tokens
│   └── word_to_idx.json         # Token-to-index mapping dictionary
├── models/
│   └── best_model.keras         # Trained LSTM captioning model checkpoint
├── notebooks/                   # Step-by-step exploratory & training notebooks
├── tests/
│   └── test_api.py              # Pytest test cases for FastAPI endpoints
├── api.py                       # FastAPI application serving /predict and /health
├── app.py                       # Streamlit frontend client
├── predict.py                   # Model loading and beam search inference logic
├── Dockerfile.api               # Container definition for FastAPI inference backend
├── Dockerfile.frontend          # Container definition for Streamlit frontend
├── docker-compose.yml           # Multi-container orchestration config
├── requirements.txt             # Backend dependencies (TensorFlow CPU, FastAPI)
└── requirements-frontend.txt    # Lightweight frontend dependencies (Streamlit, gTTS)
```

---

## Quick Start (Run Locally with Docker)

### Prerequisites
- [Docker Desktop](https://www.docker.com/) running with WSL 2 integration enabled (or native Docker engine on Linux).

### 1. Clone the Repository
```bash
git clone [https://github.com/anjXcode/VisionTalk.git](https://github.com/anjXcode/VisionTalk.git)
cd VisionTalk
```

### 2. Launch the Application Stack
```bash
docker compose up --build
```

### 3. Access the Services
- **Streamlit Web Application**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive Docs (Swagger UI)**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **API Health Endpoint**: [http://localhost:8080/health](http://localhost:8080/health)

To stop the containers:
```bash
docker compose down
```

---

## Local Development (Without Docker)

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-frontend.txt
```

### 2. Start the Inference API
```bash
uvicorn api:app --host 127.0.0.1 --port 8080 --reload
```

### 3. Start the Streamlit Frontend (In a Second Terminal)
```bash
source .venv/bin/activate
streamlit run app.py
```

---

## Automated Testing

Run the test suite locally using pytest:
```bash
pytest tests/ -v
```

Tests validate:
- Root API index and documentation endpoints.
- `/health` check status and model loading verification.
- Rejection of invalid file types (HTTP 400).
- Synthesized image inference execution and caption format validation.