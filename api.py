import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import (
    load_all_artifacts,
    generate_caption,
    generate_caption_beam_search
)

# Global model artifact containers
feat_extractor = None
caption_model = None
word_to_idx = None
idx_to_word = None
max_len = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts into memory on server startup."""
    global feat_extractor, caption_model, word_to_idx, idx_to_word, max_len
    try:
        feat_extractor, caption_model, word_to_idx, idx_to_word, max_len = load_all_artifacts()
        print("[INFO] Model artifacts successfully loaded into memory.")
    except Exception as e:
        print(f"[ERROR] Failed to load model artifacts: {e}")
    yield
    print("[INFO] Shutting down VisionTalk API...")

app = FastAPI(
    title="VisionTalk API",
    description="REST API for Deep Learning Image Caption Generation using DenseNet201 + LSTM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class CaptionResponse(BaseModel):
    caption: str
    strategy: str
    beam_width: int
    filename: str

@app.get("/")
def root():
    return {
        "message": "VisionTalk API is live!",
        "interactive_docs": "/docs",
        "alternative_docs": "/redoc",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": caption_model is not None
    }

@app.post("/predict", response_model=CaptionResponse)
async def predict_caption(
    file: UploadFile = File(...),
    beam: int = Query(default=3, ge=1, le=7, description="Beam width (1 for Greedy, >1 for Beam Search)"),
    penalty: float = Query(default=0.5, ge=0.0, le=1.0, description="Repetition penalty factor")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_img_path = temp_file.name

    try:
        if beam <= 1:
            strategy = "Greedy Search"
            caption = generate_caption(
                temp_img_path,
                feat_extractor,
                caption_model,
                word_to_idx,
                idx_to_word,
                max_len,
                repetition_penalty=penalty
            )
        else:
            strategy = f"Beam Search (k={beam})"
            caption = generate_caption_beam_search(
                temp_img_path,
                feat_extractor,
                caption_model,
                word_to_idx,
                idx_to_word,
                max_len,
                beam_index=beam,
                repetition_penalty=penalty
            )

        return {
            "caption": caption,
            "strategy": strategy,
            "beam_width": beam,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=False)