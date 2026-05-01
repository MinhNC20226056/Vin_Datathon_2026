# Datathon 2026 — Solution Overview

A two-part solution for the Datathon 2026 competition: a business-focused EDA followed by a CatBoost-based ML model that forecasts daily Revenue and COGS.

---

## Part 1 — Exploratory Data Analysis (`part2_eda.ipynb`)

The EDA is built around one central thesis: **the main business bottlenecks are margin leakage from Urban Blowout promotions and chronic stockout exposure in revenue-heavy categories** — not traffic generation or shipping SLA.

### Dataset
13 tables covering orders, order items, products, customers, promotions, geography, payments, shipments, returns, reviews, inventory, web traffic, and a daily sales summary.

### Key Findings

| # | Area | Finding |
|---|------|---------|
| 1 | **Margin leakage** | 100% of the top 0.5% `COGS / Revenue` spike days fall inside **Urban Blowout** campaign windows, where COGS/Revenue reaches **1.44×–1.57×** (gross margin of −44% to −57%). |
| 2 | **Promotions** | Promotions touch **38.66%** of item lines, but promo orders carry **20.5% lower AOV** than non-promo orders. Urban Blowout and Year-End Sale show negative seasonal revenue lift with severe margin compression. |
| 3 | **Category mix** | **Streetwear** drives ~**79.9%** of revenue but holds only **13.24%** margin rate. **GenZ** is the most margin-efficient category (19.13%) but contributes only 2.1% of revenue. |
| 4 | **Retention** | Month-1 repeat purchase rate is only **3.55%** and stabilises around 3.23% by month 6. The biggest drop-off happens immediately after the first order. |
| 5 | **Inventory** | Overall product-month stockout rate is **67.34%** — the primary operational bottleneck. Streetwear stockout alone represents the largest share of potential lost revenue. |
| 6 | **Fulfillment / Ratings** | Ship lag averages 1.5 days (max 3), delivery averages 4.5 days (max 7). Rating spread across categories is only 0.017 stars. Neither is a primary lever. |

### Prescriptive Recommendations
- Hard-block any campaign with projected `COGS / Revenue ≥ 1.0x`; add a minimum gross-margin guardrail of 10–15% before campaign launch.
- Rebuild promo mechanics around minimum basket thresholds and category-specific discount caps.
- Prioritise Streetwear replenishment first, then selectively restock GenZ and Outdoor for margin lift.
- Trigger lifecycle campaigns (reorder reminders, category vouchers) within 30–60 days after the first purchase.

---

## Part 2 — ML Forecasting Model (`final_solution.ipynb`)

A self-contained Kaggle notebook that predicts daily **Revenue** and **COGS** for 2023–2024.

### Pipeline Summary

```
Historical sales (2013–2022)
        │
        ▼
Seasonal baseline  ──  monthly × daily (month, day) pattern
        │
        ▼
CatBoost residual model  ──  trained on log1p(target) − log1p(baseline)
        │
        ▼
Recursive forecasting  ──  2023 predictions used as history for 2024
        │
        ▼
Recent-regime calibration  ──  level damping inferred from latest annual rebound
        │
        ▼
submission.csv
```

### Features
- **Calendar**: year, month, day, day-of-week, day-of-year, week-of-year
- **Lag & rolling**: prior-year same-day values, rolling means
- **Event windows**: campaign and holiday indicators
- **Categorical**: month name, weekday name, quarter

### Validation
Time-aware hold-out on **2021–2022** (the period closest to the recent demand regime). Metrics reported: MAE, RMSE, R².

### Explainability
- **CatBoost feature importance** (`PredictionValuesChange`): top drivers are trend/recent-regime level, weekday interactions, yearly lag, month-level demand, and campaign windows.
- **Built-in SHAP values**: recomputed on 2024 forecast rows without an external `shap` dependency; confirms the same feature ranking with directional decomposition.

### Output
| File | Description |
|------|-------------|
| `submission.csv` | Kaggle submission file (Date, Revenue, COGS) |
| `submission_final.csv` | Research artifact with full forecast series |
| `final_validation_recent_cv.csv` | Validation metrics per target |
| `final_feature_importance.csv` | Feature importance table |
| `final_shap_summary.csv` | Mean absolute SHAP values per feature per target |

---

## Repository Structure

```
.
├── part2_eda.ipynb                      # Part 1 — Business EDA
├── final_solution.ipynb                 # Part 2 — ML model & Kaggle submission
├── datathon-hackathon-mcq-phase-1.ipynb # Phase 1 MCQ notebook
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## How to Run

### Prerequisites
- Python 3.10 or higher
- The competition dataset CSV files placed in the same directory as the notebooks (or update the path variable inside each notebook)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the EDA notebook (Part 1)

```bash
jupyter notebook part2_eda.ipynb
```

Execute all cells from top to bottom (Kernel → Restart & Run All).  
The notebook produces inline charts summarising the business findings documented above.

### 3. Run the ML forecasting notebook (Part 2 — generates submission)

```bash
jupyter notebook final_solution.ipynb
```

Execute all cells from top to bottom (Kernel → Restart & Run All).

**Output files produced in the working directory:**

| File | Description |
|------|-------------|
| `submission.csv` | Kaggle submission file (Date, Revenue, COGS) |
| `submission_final.csv` | Full forecast series (research artifact) |
| `final_validation_recent_cv.csv` | Validation metrics per target |
| `final_feature_importance.csv` | Feature importance table |
| `final_shap_summary.csv` | Mean absolute SHAP values per feature per target |

### 4. (Optional) Run without a browser — command-line execution

```bash
jupyter nbconvert --to notebook --execute final_solution.ipynb --output final_solution_executed.ipynb
```

This runs the notebook headlessly and saves the executed version with all outputs needed.
