"""
DocPredic FastAPI
REST API for doctor department prediction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import joblib

from src.config import MODELS_DIR, LABEL_DIR, PROCESSED_DATA_DIR
from src.inference import Predictor
from src.preprocessing import preprocess_text

app = FastAPI(
    title="DocPredic API",
    description="Doctor Department/Specialty Prediction from Patient Symptoms",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[Predictor] = None


class SymptomInput(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Patient's symptom/problem description in natural language",
        examples=["I have been having severe headaches for two days, I feel dizzy and I have vomited twice."],
    )


class Prediction(BaseModel):
    department: str
    confidence: float


class Explanation(BaseModel):
    matched_symptoms: List[str] = []
    denied_symptoms: List[str] = []
    ml_top: Optional[str] = None
    knowledge_top: Optional[str] = None
    models_agree: Optional[bool] = None
    margin_top2: Optional[float] = None


class PredictionResponse(BaseModel):
    input_text: str
    processed_text: str
    top_prediction: str
    top_confidence: float
    predictions: List[Prediction]
    model_type: Optional[str] = None
    low_confidence: Optional[bool] = None
    triage_note: Optional[str] = None
    explanation: Optional[Explanation] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def load_model():
    global predictor
    try:
        # Prefer the fused high-accuracy system (ensemble + knowledge scorer).
        try:
            predictor = Predictor.from_fused_artifacts(
                models_dir=MODELS_DIR,
                processed_dir=PROCESSED_DATA_DIR,
                label_dir=LABEL_DIR,
            )
            print("Fused high-accuracy model loaded successfully")
            return
        except Exception as e:
            print(f"Fused artifacts unavailable ({e}); trying legacy model")

        model_path = MODELS_DIR / "best_sklearn_model.joblib"
        label_path = LABEL_DIR / "label_encoder.joblib"
        vectorizer_path = PROCESSED_DATA_DIR / "tfidf_vectorizer.joblib"

        if model_path.exists() and label_path.exists():
            model = joblib.load(model_path)
            label_encoder = joblib.load(label_path)
            vectorizer = joblib.load(vectorizer_path) if vectorizer_path.exists() else None

            predictor = Predictor(
                model=model,
                label_encoder=label_encoder,
                vectorizer=vectorizer,
                model_type="sklearn",
            )
            print("Model loaded successfully")
        else:
            print(f"Model not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: SymptomInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.predict(input_data.text)
        expl = result.get("explanation") or {}
        return PredictionResponse(
            input_text=result["input_text"],
            processed_text=result["processed_text"],
            top_prediction=result["top_prediction"],
            top_confidence=result["top_confidence"],
            predictions=[
                Prediction(department=p["department"], confidence=p["confidence"])
                for p in result["predictions"]
            ],
            model_type=result.get("model_type", "sklearn"),
            low_confidence=result.get("low_confidence"),
            triage_note=result.get("triage_note"),
            explanation=Explanation(
                matched_symptoms=expl.get("matched_symptoms", []),
                denied_symptoms=expl.get("denied_symptoms", []),
                ml_top=expl.get("ml_top"),
                knowledge_top=expl.get("knowledge_top"),
                models_agree=expl.get("models_agree"),
                margin_top2=expl.get("margin_top2"),
            ) if expl else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(texts: List[SymptomInput]):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = predictor.predict_batch([t.text for t in texts])
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
