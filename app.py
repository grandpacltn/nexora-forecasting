import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).parent

model = joblib.load(BASE / "models" / "nexora_model.joblib")

with open(BASE / "models" / "features.json") as f:
    FEATURES = json.load(f)["features"]

history = pd.read_csv(BASE / "outputs" / "features_daily.csv", parse_dates=["Date"])

START = date(2024, 1, 1)
LAST = history["Date"].max().date()
MAX_AHEAD = LAST + timedelta(days=7)
CLOSED_SUNDAY = {1, 2, 4}

app = FastAPI(title="Nexora Care Flow Forecast")


class ForecastRequest(BaseModel):
    clinic_id: int = Field(
        ...,
        ge=1,
        le=4,
        examples=[2],
        description="1 Riverside, 2 Lakeside, 3 Downtown Express, 4 Hillcrest",
    )
    forecast_date: date = Field(
        ...,
        ge=date(2024, 1, 1),
        le=date(2026, 1, 7),
        examples=["2026-01-05"],
        description="Forecast dates: 2026-01-01 to 2026-01-07",
    )


def build_features(clinic_id, d):
    h = history[history["ClinicID"] == clinic_id].set_index("Date")["Appointments"]

    def lag(n):
        return h.get(pd.Timestamp(d - timedelta(days=n)), float("nan"))

    window = h[
        (h.index >= pd.Timestamp(d - timedelta(days=34)))
        & (h.index <= pd.Timestamp(d - timedelta(days=7)))
    ]
    flu = int(history.loc[history["Month"] == d.month, "IsFluSeason"].iloc[-1])

    values = {
        "ClinicID": clinic_id,
        "DayOfWeek": d.weekday(),
        "Month": d.month,
        "IsHoliday": 1 if (d.month, d.day) == (1, 1) else 0,
        "DayAfterHoliday": 1 if (d.month, d.day) == (1, 2) else 0,
        "IsFluSeason": flu,
        "DaysSinceStart": (d - START).days,
        "Lag7": lag(7),
        "Lag14": lag(14),
        "Lag28": lag(28),
        "Roll28": window.mean() if len(window) else float("nan"),
    }
    return pd.DataFrame([values])[FEATURES]


@app.get("/health")
def health():
    return {"status": "ok", "history_rows": len(history), "history_ends": str(LAST)}


@app.post("/predict")
def predict(req: ForecastRequest):
    d = req.forecast_date

    if d <= LAST:
        row = history[
            (history["ClinicID"] == req.clinic_id)
            & (history["Date"] == pd.Timestamp(d))
        ]
        if row.empty:
            raise HTTPException(404, "This clinic is closed on that day.")
        pred = float(model.predict(row[FEATURES])[0])
        return {
            "type": "backtest",
            "clinic_id": req.clinic_id,
            "date": str(d),
            "predicted_appointments": round(max(pred, 0), 1),
            "actual_appointments": int(row["Appointments"].iloc[0]),
        }

    if d > MAX_AHEAD:
        raise HTTPException(
            400, f"Forecasts cover up to {MAX_AHEAD}. Refresh the history to go further."
        )
    if d.weekday() == 6 and req.clinic_id in CLOSED_SUNDAY:
        raise HTTPException(404, "This clinic is closed on Sundays.")

    pred = float(model.predict(build_features(req.clinic_id, d))[0])
    return {
        "type": "forecast",
        "clinic_id": req.clinic_id,
        "date": str(d),
        "predicted_appointments": round(max(pred, 0), 1),
    }