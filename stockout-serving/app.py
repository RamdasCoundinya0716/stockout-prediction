from fastapi import FastAPI, Header, HTTPException
from schema import PredictRequest, PredictResponse
import joblib
import pandas as pd

API_KEY = "ramdas-stockout-prediction-!@#$!@#!@#"   # change this to something strong
app = FastAPI(title="Stockout Risk Prediction API", version="1.0")

MODEL_PATH = "stockout_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    "price", "promo", "daily_sales", "inventory",
    "sales_lag_1", "sales_lag_3", "sales_lag_7",
    "inv_lag_1", "promo_lag_1",
    "sales_roll7_avg", "sales_roll7_sum", "sales_roll14_avg",
    "promo_roll7_sum",
    "inv_to_sales_ratio", "low_inventory_flag"
]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    row = req.model_dump()
    X = pd.DataFrame([row], columns=FEATURE_COLS)

    score = float(model.predict_proba(X)[:, 1][0])
    label = 1 if score >= 0.5 else 0

    return PredictResponse(
        stockout_risk_score=score,
        stockout_risk_label=label
    )
