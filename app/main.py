import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from app.schemas import CreditProfile, RiskAssessmentResponse

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
bundle = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = ARTIFACTS_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError("Artifacts missing. Run 'python src/train.py' first.")
    
    bundle["model"] = joblib.load(model_path)
    bundle["explainer"] = joblib.load(ARTIFACTS_DIR / "explainer.joblib")
    bundle["features"] = joblib.load(ARTIFACTS_DIR / "features.joblib")
    yield
    bundle.clear()

app = FastAPI(
    title="CreditRisk-XAI: Explainable Credit Underwriting API",
    description="High-precision risk prediction with TreeSHAP legal adverse-action attribution",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "model" in bundle}

@app.post("/assess-risk", response_model=RiskAssessmentResponse)
def assess_risk(profile: CreditProfile):
    start_time = time.perf_counter()
    model = bundle.get("model")
    explainer = bundle.get("explainer")
    features = bundle.get("features")
    
    if not model or not explainer:
        raise HTTPException(status_code=503, detail="Model artifacts not initialized")
    
    input_data = pd.DataFrame([profile.model_dump()])[features]
    
    probs = model.predict_proba(input_data)[0]
    default_prob = float(probs[1])
    
    threshold = 0.55
    risk_level = "High" if default_prob >= threshold else "Low"
    
    shap_vals = explainer.shap_values(input_data)[0]
    factors = [
        {
            "feature": feat,
            "impact_score": round(abs(float(val)), 4),
            "direction": "Increases Risk" if val > 0 else "Decreases Risk"
        }
        for feat, val in zip(features, shap_vals)
    ]
    factors.sort(key=lambda x: x["impact_score"], reverse=True)
    
    latency = round((time.perf_counter() - start_time) * 1000, 2)
    
    return RiskAssessmentResponse(
        risk_level=risk_level,
        default_probability=round(default_prob, 4),
        decision_threshold=threshold,
        top_adverse_factors=factors[:3],
        latency_ms=latency
    )