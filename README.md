# Forecasting Appointment Volume for Nexora Care Flow

A time-series forecasting project that predicts daily appointment volume for four clinics, so administrators can plan staffing before each week begins instead of reacting to busy days after they start.

**Author:** Mathew Ogoh Odinaka
**Project:** Amdari data science project, "Forecasting Appointment Volume for Nexora Care Flow"

---

## 1. Business problem

Nexora Care Flow is scheduling and clinic-operations software for small and mid-sized clinics. Appointment demand at the pilot clinics goes up and down, and administrators plan staff and rooms from gut feel and last month's rough average. They only learn a day is busy once it is underway.

| Impact | What it means |
|---|---|
| Efficiency | Staff are overwhelmed on unexpected busy days |
| Cost | Overstaffing on quiet days, overtime on surges |
| Patient experience | Longer waits and rushed visits |

**Goal:** a validated forecast of clinic-level appointment volume, scored against a naive baseline with MAPE and RMSE, turned into plain staffing guidance.

**Out of scope (by design):** outside data, databases, deployment, dashboards, automated retraining, and automatic scheduling changes. The guidance is advisory only.

---

## 2. Data

- One flat file exported from Nexora Care Flow: `AppointmentRecords.csv`, one row per appointment.
- **Raw:** 129,353 rows and 15 columns. **Cleaned:** 129,352 rows and 17 columns (added `Date` and `DayOfWeek`).
- **Period:** 1 January 2024 to 31 December 2025. The project brief describes one quarter of data, but the file contains two full years. This is recorded in the data-quality log.
- **Clinics:** Riverside, Lakeside, Downtown Express and Hillcrest (24 providers, 7,109 patients).
- The raw data is **not included** in this repository. The `data/` folder is listed in `.gitignore`.

---

## 3. Repository structure

```
nexora-forecasting/
├── notebooks/
│   ├── datacleaning.ipynb   Step 1: loading, profiling, cleaning, data-quality log, export
│   ├── eda.ipynb            Step 2: exploratory data analysis
│   └── features.ipynb       Steps 3 to 5: features, split, baseline, model, evaluation, staffing flags
├── outputs/
│   ├── data_quality_log.csv   Every check, finding and action from cleaning
│   └── features_daily.csv     Engineered feature set (daily counts per clinic, no patient identifiers)
├── data/                    Raw and cleaned appointment files (not tracked by Git)
├── .gitignore
├── LICENSE
└── README.md
```

---

## 4. Method

### Step 1: Data cleaning (`datacleaning.ipynb`)

| Check | Finding | Action |
|---|---|---|
| Missing values | None, except `HolidayName` (128,414 empty) | Kept: empty exactly where `IsHoliday` is False |
| Exact duplicate rows | 0 | None needed |
| Duplicate `AppointmentID` | 0 | None needed |
| Same details, different ID | 1 pair (IDs 625580 and 625614, Hillcrest, 2025-10-29 10:30) | Removed 625614, kept 625580 |
| Date format | All 129,353 values in one format, no conversion failures | Converted to datetime |
| Categorical spellings | No variants | None needed |
| Lead time | 0 to 60 days, no negatives | None needed |
| Clinic closures | Sundays closed at Riverside, Lakeside and Hillcrest; Hillcrest opened on 29 June 2024 | Closed days left out of the daily counts, not filled with zeros |
| Lowest-volume days | All public holidays | Kept: real demand dips the model must learn |

An untouched raw copy (`df_raw`) was kept during cleaning.

### Step 2: Exploratory analysis (`eda.ipynb`)

Questions that guided the analysis:

1. What explains the rise from 2024 to 2025, and was it consistent across months?
2. Which weekdays and months run above normal?
3. Do some clinics consistently handle more appointments than others?
4. How much do cancellations and no-shows contribute?
5. Does booking lead time relate to attending, cancelling or missing?

Holiday impact and season effects were analysed because the project guide asks for them.

| Driver | Finding | Strength |
|---|---|---|
| Holidays | About 80% to 85% below a normal day; about 27% above normal the day after; back to normal within 1 to 2 days (recovery values read from a chart, so approximate) | Very strong |
| Clinic level | Lakeside averages 73 appointments per open day, Downtown Express 34; Lakeside led in all 24 months | Strong |
| Closures | Sundays closed at three clinics; Hillcrest opened 29 June 2024 | Strong |
| Day of week | Tuesday 19% to 23% above each clinic's own average; Saturday 16% to 67% below | Strong |
| Season | Flu season 14% to 19% above average at three clinics (partly overlaps winter months and growth, so not a pure illness effect) | Moderate |
| Month | December +12% to +17% | Moderate |
| Growth | Total +32.1%, but about 68% of the extra volume is Hillcrest opening. The other three clinics grew 12.0%, and growth came from existing patients booking more visits (9.6 to 10.6 per patient) | Moderate |
| Lead time | No-shows rise from about 8% to 17% for bookings made more than two weeks ahead | Weak overall |
| Type, channel | The share of bookings not completed stays near 15.7% everywhere | Negligible |

### Step 3: Feature engineering (`features.ipynb`)

The data is aggregated to **one row per clinic per open day** (2,457 rows).

| Feature | Meaning |
|---|---|
| `Lag7`, `Lag14`, `Lag28` | Volume 7, 14 and 28 calendar days earlier |
| `Roll28` | 28-day rolling average, ending 7 days before each date |
| `DayOfWeek`, `Month` | Calendar position |
| `IsHoliday`, `DayAfterHoliday` | Holiday indicators |
| `IsFluSeason` | Flu-season indicator |
| `DaysSinceStart` | Trend counter |
| `ClinicID` | Clinic identifier |

All lags are 7 or more days, so the features suit a **one-week-ahead** forecast.

**Time-ordered split, with no shuffling:** training is 1 January 2024 to 30 September 2025 (2,128 rows). Test is 1 October to 31 December 2025 (329 rows, 92 days).

### Step 4: Model

- **Baseline:** the same weekday 28 days earlier (the guide's "same week last month").
- **Model:** scikit-learn `HistGradientBoostingRegressor`, with `ClinicID` and `DayOfWeek` treated as categorical, `random_state=42`, one pooled model for all clinics, otherwise default settings.
- **Tried and not adopted:** adding the same weekday last year (`Lag364`). It did not help on a validation window (July to September 2025), so the simpler model was kept.

---

## 5. Results

### Test period, October to December 2025

| Measure | Baseline | Model |
|---|---|---|
| MAPE, all days | 60.59% | **17.85%** |
| RMSE, all days | 23.21 | **11.60** |
| MAPE, normal days (293 rows) | 20.13% | **15.14%** |
| RMSE, normal days | 16.84 | **11.92** |
| MAPE, holiday and day-after (36 rows) | 389.85% | **39.97%** |
| Average weekly error, 48 clinic-weeks | 12.8% | **9.6%** |

**Success criterion met:** the model beats the baseline on both MAPE and RMSE on the test period.

- The baseline's very high MAPE comes mostly from holidays, where it copies a normal day onto a day with almost no appointments. On normal days the model is still about 25% better on MAPE and 29% better on RMSE.
- Same weekday last week scores worse than the baseline on normal days (MAPE 21.95%, RMSE 18.91), so the baseline is not unusually weak.

### Per clinic (test period)

| Clinic | Baseline MAPE | Model MAPE | Baseline RMSE | Model RMSE |
|---|---|---|---|---|
| Downtown Express | 61.24% | 19.53% | 16.64 | 8.84 |
| Hillcrest | 45.34% | 15.62% | 23.59 | 12.83 |
| Lakeside | 62.51% | 14.42% | 29.73 | 13.55 |
| Riverside | 73.16% | 21.57% | 21.91 | 11.00 |

### Three rolling windows

| Window | Baseline MAPE | Model MAPE | Baseline RMSE | Model RMSE | Model bias |
|---|---|---|---|---|---|
| Apr to Jul 2025 | 48.71% | 18.94% | 18.28 | 10.80 | -1.05 |
| Jul to Oct 2025 | 42.21% | 17.13% | 19.81 | 11.36 | -1.84 |
| Oct 2025 to Jan 2026 | 60.59% | 17.85% | 23.21 | 11.60 | -4.05 |

Each window was scored by a model trained only on earlier dates.

### Feature importance (permutation, test set)

`IsHoliday` (12.30) > `DayOfWeek` (7.78) > `Roll28` (4.17) > `ClinicID` (3.19) > `Lag7` (1.99) > `IsFluSeason` (1.37). The trend counter `DaysSinceStart` scored 0.00, so the model did not use it.

---

## 6. Staffing guidance

Each clinic has a "normal week": the **median weekly total over its last 12 training weeks** (7 July to 28 September 2025). A forecast week is flagged **HIGH** if it is more than 15% above normal and **LOW** if it is more than 15% below. The 15% threshold was set before looking at the results and sits above the model's average weekly error.

| Flags built from | Type | Flagged | Correct | Real weeks | Precision | Recall |
|---|---|---|---|---|---|---|
| Model | HIGH | 4 | 3 | 12 | 75% | 25% |
| Model | LOW | 7 | 3 | 4 | 43% | 75% |
| Baseline | HIGH | 3 | 0 | 12 | 0% | 0% |
| Baseline | LOW | 3 | 0 | 4 | 0% | 0% |

**How to use it**
- Treat a HIGH flag as reliable: when the model said HIGH, it was right 75% of the time.
- Do not rely on the absence of a HIGH flag. The model caught only 3 of 12 surge weeks, mostly missing December surges.
- Holiday weeks (Thanksgiving, Christmas) were flagged LOW, which is right in direction, but the drop at some clinics was smaller than forecast.
- Treat the forecast as a minimum in peak periods such as December, and add a buffer.

---

## 7. Limitations

- **Under-forecasting.** The model leans low: an average error of -4.05 per clinic-day (October -1.7, November -2.5, December -7.9), and the lean grew across the three windows. 7 of the 8 worst weekly misses were under-forecasts, 5 of them in December. Possible causes (not tested): December appears only once in the training data, and the trend counter is unused.
- **Weekly totals.** The baseline was slightly better at Downtown Express (8.0% vs 10.0% weekly error) and Hillcrest (9.0% vs 10.1%) over the 12 test weeks.
- **One test window** of 92 days, containing holidays that the model has seen only once.
- **Two years of history**, and about 18 months for Hillcrest.
- **No outside data** such as weather or illness feeds.
- The data cannot explain why patients visit more often or miss appointments.
- Lags are 7 or more days, so the model fits a one-week-ahead forecast, not a longer horizon.

---

## 8. Improvement roadmap

**Weekly (about 15 minutes)**
- Compare last week's forecast with actual volume per clinic. Record the weekly error % and the average error (forecast minus actual).
- Update each clinic's normal week from its last 12 weeks.

**Monthly**
- Retrain on all data to date and rescore on the most recent weeks, using the same baseline.

**Proposed refresh triggers (to agree with the team)**
- Retrain early if weekly error is above 15% for three weeks in a row, or if the average error stays below -5% of a normal week.

**Next steps**
- Correct the low lean (for example, adjust forecasts using the recent average error) and test it on a validation window first.
- Compare with Prophet and per-clinic models. Prophet and statsmodels are installed but not yet used.
- Add a third year of data, especially a second December, and more history for Hillcrest.
- Consider outside data (weather, illness feeds) in a later phase.

---

## 9. How to run

**Requirements:** Python 3.10 and VS Code with Jupyter support (or Jupyter Notebook).

```bash
git clone https://github.com/grandpacltn/nexora-forecasting.git
cd nexora-forecasting
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install pandas numpy matplotlib seaborn statsmodels prophet scikit-learn jupyter ipykernel
```

1. Create a `data/` folder and place the raw `AppointmentRecords.csv` in it. The file is not included in this repository.
2. Run the notebooks in this order, choosing the `.venv` kernel and using **Run All**:
   1. `notebooks/datacleaning.ipynb`: writes `data/AppointmentRecords_clean.csv` and `data/data_quality_log.csv`
   2. `notebooks/eda.ipynb`: reads the clean file
   3. `notebooks/features.ipynb`: builds the features, trains and scores the model

Results are reproducible: the model uses `random_state=42`.

---

## 10. Tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter, VS Code, Git and GitHub.
