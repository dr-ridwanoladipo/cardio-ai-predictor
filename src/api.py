# uvicorn src.api:app --reload
"""
🩺 Clinical Heart Disease AI - FastAPI REST API
World-class API for serving heart-disease predictions and SHAP explainability.

Author: Ridwan Oladipo, MD | AI Specialist
"""

# ── Imports ────────────────────────────────────────────────────────────────────
from datetime import datetime
import logging
import time
import traceback
from typing import Any, Dict, List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.model import initialize_model, predict_heart_disease, predictor

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y | %I:%M%p",
    handlers=[logging.FileHandler("outputs/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# =========================
# Rate Limiting
# =========================
limiter = Limiter(key_func=get_remote_address)

# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Clinical Heart Disease AI API",
    description="Production API for heart disease prediction & clinical-grade explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} in {(time.time()-start)*1000:.2f} ms")
    return response

# =========================
# State & util
# =========================
model_loaded = False
startup_time = None
def current_time_iso():  # keep original def style
    return datetime.now().isoformat()

# =========================
# Schemas
# =========================
class PatientInput(BaseModel):
    age:      int   = Field(..., ge=18,  le=101, description="Patient age in years (18-100)")
    sex:      int   = Field(..., ge=0,   le=1,   description="Sex (0=Female, 1=Male)")
    cp:       int   = Field(..., ge=0,   le=3,   description="Chest pain type")
    trestbps: int   = Field(..., ge=80,  le=301, description="Resting blood pressure (80-300 mmHg)")
    chol:     int   = Field(..., ge=100, le=601, description="Cholesterol (100-600 mg/dl)")
    fbs:      int   = Field(..., ge=0,   le=1,   description="Fasting blood sugar >120 mg/dl")
    restecg:  int   = Field(..., ge=0,   le=2,   description="Resting ECG result")
    thalach:  int   = Field(..., ge=60,  le=221, description="Max heart rate achieved")
    exang:    int   = Field(..., ge=0,   le=1,   description="Exercise induced angina")
    oldpeak:  float = Field(..., ge=0.0, le=10.1,description="ST depression")
    slope:    int   = Field(..., ge=0,   le=2,   description="Slope of ST segment")
    ca:       int   = Field(..., ge=0,   le=3,   description="Number of major vessels")
    thal:     int   = Field(..., ge=1,   le=3,   description="Thalassemia status")

    @field_validator("thalach")
    @classmethod
    def validate_hr_vs_age(cls, v, info):
        age = info.data.get("age")
        if age and v > (220 - age) + 20:
            raise ValueError(f"Heart rate {v} unusually high for age {age}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 0, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
            }
        }

class PredictionResponse(BaseModel):
    probability: float
    risk_class: str
    clinical_summary: str
    timestamp: str

class SHAPResponse(BaseModel):
    shap_values: List[float]
    top_features: List[Dict[str, Any]]
    feature_names: List[str]
    timestamp: str

class PositionsResponse(BaseModel):
    feature_positions: Dict[str, Dict[str, Any]]
    guideline_categories: Dict[str, str]
    timestamp: str

class MetricsResponse(BaseModel):
    model: str
    roc_auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    confusion_matrix: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    startup_time: str
    timestamp: str
    version: str

# =========================
# Startup / shutdown
# =========================
@app.on_event("startup")
async def startup_event():
    global model_loaded, startup_time
    startup_time = current_time_iso()
    logger.info("Starting API…")
    try:
        model_loaded = initialize_model()
        logger.info("Model ready" if model_loaded else "Model failed to load")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API…")

# =========================
# Routes
# =========================
@app.get("/", summary="Clinical Heart Disease AI API Overview", tags=["App Info"])
async def root():
    return {
        "app": "Clinical Heart Disease AI",
        "purpose": "Machine learning API for coronary artery disease (CAD) risk prediction with full clinical explainability.",
        "model": {
            "type": "XGBoost (optimized)",
            "explainability": "SHAP",
            "performance": {"roc_auc": 0.91, "sensitivity": 0.97, "specificity": 0.71},
        },
        "author": "Ridwan Oladipo, MD | AI Specialist",
        "version": "1.0.0",
        "documentation": "/docs",
    }

@app.get("/health", response_model=HealthResponse, summary="Service health check", tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok" if predictor.model else "error",
        model_loaded=bool(predictor.model),
        version="1.0.0",
        startup_time=startup_time,
        timestamp=current_time_iso(),
    )

@app.post("/predict", response_model=PredictionResponse, summary="Predict CAD risk", tags=["Predictions"])
@limiter.limit("5/minute")
async def predict(request: Request, patient: PatientInput):
    if not model_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        logger.info(f"Predict: age={patient.age}, sex={patient.sex}")
        results = predict_heart_disease(patient.dict())
        p = results["prediction"]
        return PredictionResponse(
            probability=p["probability"],
            risk_class=p["risk_class"],
            clinical_summary=p["clinical_summary"],
            timestamp=current_time_iso(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed")

@app.post("/shap", response_model=SHAPResponse, summary="Get SHAP explainability", tags=["Explainability"])
@limiter.limit("5/minute")
async def get_shap_explanation(request: Request, patient: PatientInput):
    if not predictor.model:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        logger.info(f"SHAP: age={patient.age}")
        results = predict_heart_disease(patient.dict())
        e = results["explanations"]
        shap_vals = [float(v) for v in e["shap_values"][0]]
        top_feats = [
            {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in feat.items()}
            for feat in e["top_features"]
        ]
        return SHAPResponse(
            shap_values=shap_vals,
            top_features=top_feats,
            feature_names=results["model_info"]["feature_names"],
            timestamp=current_time_iso(),
        )
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="SHAP explanation failed")

@app.post("/positions", response_model=PositionsResponse, summary="Patient cohort comparisons & guidelines", tags=["Patient Comparisons"])
@limiter.limit("5/minute")
async def get_feature_positions_route(request: Request, patient: PatientInput):
    if not model_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        logger.info(f"Positions: age={patient.age}")
        results = predict_heart_disease(patient.dict())
        c = results["comparisons"]
        return PositionsResponse(
            feature_positions=c["feature_positions"],
            guideline_categories=c["guideline_categories"],
            timestamp=current_time_iso(),
        )
    except Exception as e:
        logger.error(f"Position analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Position analysis failed")

@app.get("/metrics", response_model=MetricsResponse, summary="Model performance metrics", tags=["Model Metrics"])
async def get_model_metrics():
    if not model_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        logger.info("Metrics requested")
        m = predictor.get_metrics()
        return MetricsResponse(
            model=m["model"],
            roc_auc=m["roc_auc"],
            accuracy=m["accuracy"],
            sensitivity=m["sensitivity"],
            specificity=m["specificity"],
            ppv=m["ppv"],
            npv=m["npv"],
            confusion_matrix=m["confusion_matrix"],
        )
    except Exception as e:
        logger.error(f"Metrics retrieval error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Metrics retrieval failed")

# =========================
# Exception handlers
# =========================
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(status_code=422, content={"detail": f"Validation error: {exc}", "time": current_time_iso()})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Unexpected error occurred", "time": current_time_iso()})

# =========================
# Uvicorn entry
# =========================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")