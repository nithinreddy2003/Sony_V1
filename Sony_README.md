# German Electricity Demand — Time-Series Forecasting

A case study forecasting **German national electricity load** (Open Power System Data) using a
progression of models — classical benchmarks, SARIMA / SARIMAX, a feature-based Random Forest, and an
LSTM — compared on a **2-year (104-week) hold-out horizon**.

---

## Repository contents

| File | Description |
|------|-------------|
| [`report_sony_md.md`](report_sony_md.md) | **Primary written report** (Parts 1–8), corrected and audited. |
| [`version2_linear_regression.ipynb`](version2_linear_regression.ipynb) | Full analysis notebook (EDA → benchmarks → SARIMA/SARIMAX → RF/Linear → LSTM). |
| [`report.md`](report.md) | Earlier full report draft (V1). |
| `v3_report.md`, `v3_report_plan.md`, `v3_part7_analytical_questions.md` | v3 working drafts. |
| `v6_report.md`, `v6_report_outline.md`, `v6_part7_ans.md` | v6 working drafts. |
| `requirements.txt` | Pinned Python dependencies. |

> Figures referenced by `report_sony_md.md` live in the notebook export image folder
> (`vertopal_.../*.png`). Keep that folder next to the report when converting to PDF/Word.

---

## 1. Data

- **Source:** [Open Power System Data — time series](https://data.open-power-system-data.org/time_series/) (60-minute file, `2020-10-06`).
- **Series:** `DE_load_actual_entsoe_transparency` (German actual load, MW).
- **Window:** 1 Jan 2015 → 30 Sep 2020, aggregated to **daily** and **weekly** means (50,400 hourly → 301 weekly observations).
- **Exogenous data:** Berlin 2 m temperature from the [Open-Meteo archive API](https://archive-api.open-meteo.com/v1/archive) and German public holidays (`holidays` package).

> **Reproducibility:** the notebook downloads the OPSD file and caches it to `data/`. If you run
> offline, place the CSV in `data/` first. The Open-Meteo call and the LSTM both require internet /
> GPU on first run.

---

## 2. Pipeline (assignment Parts 1–8)

| Part | Content |
|------|---------|
| 1 | Data prep + EDA: daily/weekly aggregation, seasonal decomposition, ADF/KPSS stationarity, ACF/PACF |
| 2 | Benchmarks: Mean, Naive, **Seasonal Naive**, Drift |
| 3 | **SARIMA**: grid search `p∈[0,6]`, `d∈[0,2]`, `q∈[0,6]`; valid within-`d` AIC + parsimony; seasonal `(P,Q)` search with `D=1` at `s=52`; residual diagnostics |
| 4 | **SARIMAX**: temperature, temperature², 1-week temperature lag, holiday flag (conditional forecast) |
| 5 | **Random Forest**: recursive multi-step forecast (comparable to SARIMA) **and** a 1-step reference |
| 6 | **LSTM** (hourly): 5-config hyperparameter search, rolling vs open-loop evaluation |
| 7 | Analytical answers to the 6 assignment questions |
| 8 | Consolidated RMSE / MAE / MAPE table + comparison figures |

### Methodological notes
- **AIC is compared only within a fixed differencing order** — comparing across `d` is invalid because the likelihood is computed on the differenced series. `d=1` is justified from the stationarity tests (`d=2` over-differences).
- **Residual diagnostics** use standardized residuals with the state-space burn-in removed.
- **Fair comparison:** 1-step forecasts (which consume the actual previous week) are labelled separately and never compared with multi-step forecasts.
- Every model reports **RMSE, MAE and MAPE**.

---

## 3. Results (weekly hold-out, 104 weeks)

**Multi-step forecasts** (single forecast origin — the fair comparison):

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | Type |
|-------|----------:|---------:|---------:|------|
| **Seasonal Naive** | **3006.8** | 2318.5 | 4.41 | multi-step (best) |
| Random Forest (recursive) | 3049.8 | 2340.8 | 4.49 | multi-step (conditional) |
| SARIMAX (Temp + Holiday) | 3418.0 | 2682.5 | 5.12 | multi-step (conditional) |
| SARIMA | 3835.7 | 3113.6 | 5.91 | multi-step |
| Mean | 4397.3 | 3788.8 | 6.97 | multi-step |
| Naive | 4459.1 | 3783.2 | 6.79 | multi-step |
| LSTM (open-loop) | 4606.8 | 3967.5 | 7.41 | multi-step |
| Drift | 5118.0 | 4339.9 | 8.05 | multi-step |

**1-step / walk-forward** (reference only — see the actual previous value, so **not** comparable to the above):

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | Type |
|-------|----------:|---------:|---------:|------|
| LSTM (rolling) | 253.1 | 206.0 | 0.39 | 1-step (uses actual lag-1) |
| Random Forest (1-step) | 2551.5 | 1841.9 | 3.54 | 1-step (uses actual lag-1) |

Selected SARIMA model: **SARIMA(1,1,6)×(0,1,1)₅₂** (AIC = 1489.65; the seasonal search dropped the insignificant seasonal AR term).

### Key findings
- **No model beats the Seasonal Naive benchmark on multi-step accuracy.** German weekly demand is dominated by a stable annual cycle, so "repeat last year" is a very strong 2-year baseline; the recursive Random Forest essentially matches it (+43 MW, ≈1.4%).
- The **2020 COVID-19 demand dip** falls in the test window — an exogenous shock none of the models anticipate.
- **1-step** RF / rolling LSTM have low RMSE only because they see the actual previous value; they are not comparable to multi-step forecasts.
- **Recommended for operational use: SARIMAX** — not for top accuracy, but for native confidence intervals, interpretable temperature/holiday coefficients, exogenous-driver support, and low maintenance.
- SARIMA/SARIMAX residuals are **not perfect white noise** (significant Ljung-Box, non-normal per Shapiro-Wilk), so the Gaussian confidence intervals are approximate.

---

## 4. How to run

**Environment:** Python 3.12 (see `requirements.txt`).

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch Jupyter and run the notebook top to bottom
jupyter notebook version2_linear_regression.ipynb
```

**Prerequisites**
- **Internet enabled** on first run (downloads the OPSD CSV and calls the Open-Meteo temperature API).
- **GPU recommended** for the LSTM stage.

**Runtime:** roughly 30–60 minutes, dominated by the SARIMA grid search and the LSTM open-loop
recursive forecast (17,472 sequential steps).

---

## 5. References

- Open Power System Data — Time series (Germany, `DE`).
- Open-Meteo Historical Weather (Archive) API.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.
- Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference*, 2nd ed. Springer.
- Kong, W. et al. (2017). "Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural Network." *IEEE Transactions on Smart Grid*.
- Seabold, S. & Perktold, J. (2010). "statsmodels." *Proc. 9th Python in Science Conf.*
- Pedregosa, F. et al. (2011). "Scikit-learn." *JMLR*, 12, 2825–2830.

---

## 6. Notes & limitations

- SARIMAX / recursive-RF use **observed** future temperature → **conditional (explanatory) forecasts**, not fully operational (a real deployment needs a weather forecast). Holidays are deterministic and known in advance.
- The open-loop LSTM is shown separately because its recursive 2-year drift would otherwise distort the comparison plot; its RMSE also varies between runs (GPU non-determinism).
- The dataset ends Oct 2020, so the "compare against data collected after the forecast period" discussion uses the held-out test window (Oct 2018 – Oct 2020), which is real data recorded after the training cut-off.
