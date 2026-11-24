import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE = Path(__file__).parent

model = joblib.load(BASE / "models" / "nexora_model.joblib")

with open(BASE / "models" / "features.json") as f:
    FEATURES = json.load(f)["features"]

history = pd.read_csv(BASE / "outputs" / "features_daily.csv", parse_dates=["Date"])

app = FastAPI(title="Nexora Care Flow Forecast")


class ForecastRequest(BaseModel):
    clinic_id: int
    forecast_date: date


@app.get("/health")
def health():
    return {"status": "ok", "history_rows": len(history)}


@app.post("/predict")

def predict(req: ForecastRequest):
    row = history[
        (history["ClinicID"] == req.clinic_id)
        & (history["Date"] == pd.Timestamp(req.forecast_date))
    ]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail="No history row for this clinic and date. The clinic may be "
                   "closed that day, or the date is outside the history file.",
        )

    prediction = float(model.predict(row[FEATURES])[0])

    return {
        "clinic_id": req.clinic_id,
        "date": str(req.forecast_date),
        "predicted_appointments": round(max(prediction, 0), 1),
        "actual_appointments": int(row["Appointments"].iloc[0]),
    }