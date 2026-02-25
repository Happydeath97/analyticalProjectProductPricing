# 📊 Analytics Project 2026 – Dynamic Pricing & Demand Prediction

## 📌 Project Overview

This project analyzes dynamic pricing strategies in an online pharmacy setting.

Large online shops use automatic price adjustments to optimize revenue and margin at the product level. The goal is **not personalization**, but product-level price optimization aligned with market conditions.

The dataset consists of:

- `items.csv` → static product attributes
- `train.csv` → time-varying features and user interactions
- Over 2.5 million records
- User actions: click, basket, order

---

## 🎯 Objective

The goal of this project is to:

1. Perform **Exploratory Data Analysis (EDA)**
2. Engineer meaningful features
3. Train and compare **at least three classification models**
4. Predict whether a product is purchased
5. Provide **business interpretation and inference**
6. Present findings in a structured paper and presentation

---

## 🏗 Repository Structure

```
analyticalProjectProductPricing/
├── data/
│   ├── raw/
│   │   ├── items.csv
│   │   └── train.csv
│   └── processed/
├── notebooks/
├── scripts/
├── src/
│   ├── __init__.py
│   └── main.py
├── tests/
├── requirements.txt
├── README.md
└── 2026AnalyticsProject.pdf
```

---

## ⚙️ Environment Setup

### Create Virtual Environment

**Windows (PowerShell)**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import pandas, sklearn, lightgbm; print('Environment OK')"
```

---

## Running the Project

Run the main script:

```bash
python src/main.py
```
---

## Methodology

### Data Understanding
- Merge `items.csv` and `train.csv` via `pid`
- Inspect missing values
- Analyze distribution of `click` → `basket` → `order`
- Examine price and competitor price distributions

### Feature Engineering
- Price difference: `price - competitorPrice`
- Price ratio: `price / rrp`
- Campaign effects (`campainIndex`, `adFlag`)
- Availability encoding
- Categorical encoding (`manufacturer`, `group`, `unit`, `pharmForm`, etc.)
- Interaction effects (e.g., price gap × availability)

### Modeling (Minimum 3 Algorithms)
- Logistic Regression (baseline)
- Random Forest
- LightGBM (recommended for large tabular data)

### Evaluation
- ROC-AUC
- Precision / Recall
- Confusion Matrix
- Feature Importance
- SHAP analysis (model explainability)

### Business Interpretation
- Price sensitivity signals (impact of price gap vs competitors)
- Campaign effectiveness
- Availability effect on conversion
- Revenue implications

---

## Dataset Variables

### items.csv

| Variable | Description | Value Range |
|---|---|---|
| `pid` | Product number | Natural number |
| `manufacturer` | Manufacturer (anonymized) | Positive whole number |
| `group` | Product group | String (capital letters + numbers) |
| `content` | Package content | Positive float or `NxN` format |
| `unit` | Unit | String of capital letters |
| `pharmForm` | Dosage form | Three-digit string of capital letters |
| `genericProduct` | Generic flag | {0, 1} |
| `salesIndex` | Dispensing regulation code | Natural number |
| `category` | Main shop category | Natural number |
| `campainIndex` | Action label | {A, B, C} |
| `rrp` | Reference price | Positive decimal number |

### train.csv

| Variable | Description | Value Range |
|---|---|---|
| `lineID` | Unique key for user action | Natural number |
| `day` | Day in the observed period | Natural number |
| `pid` | Product number | Natural number |
| `adFlag` | Advertising/campaign flag | {0, 1} |
| `availability` | Availability status | {1, 2, 3, 4} |
| `competitorPrice` | Lowest competitor price | Positive decimal number |
| `click` | Click flag | {0, 1} |
| `basket` | Basket flag | {0, 1} |
| `order` | Order flag | {0, 1} |
| `price` | Product price | Positive decimal number |
| `revenue` | Revenue | Positive decimal number |

---

## 📈 Business Perspective

**Central business question:**
> How does product price positioning relative to competitors influence purchase probability and revenue?

Results should support:
- Improved pricing strategies
- Better revenue optimization decisions
- Understanding of campaign impact

---

## Milestones

### Phase 0 – Setup
- [x] Initialize repository structure
- [x] Create virtual environment
- [x] Add `requirements.txt`
- [x] Add project assignment PDF
- [x] Create README

### Phase 1 – Data Exploration
- [ ] Load datasets
- [ ] Inspect missing values
- [ ] Analyze class imbalance (`order` vs non-order)
- [ ] Explore price distributions and outliers
- [ ] Funnel analysis (`click` → `basket` → `order`)
- [ ] Correlation / dependency analysis

### Phase 2 – Feature Engineering
- [ ] Merge static and dynamic data on `pid`
- [ ] Handle missing values
- [ ] Encode categorical variables
- [ ] Create pricing features (difference, ratio, discount proxy)
- [ ] Create time-based features from `day`
- [ ] Create interaction terms
- [ ] Save processed dataset(s) to `data/processed/`

### Phase 3 – Modeling
- [ ] Create train/validation split strategy (time-aware if needed)
- [ ] Train Logistic Regression baseline
- [ ] Train Random Forest model
- [ ] Train LightGBM model
- [ ] Compare model performance with consistent metrics
- [ ] Select best model

### Phase 4 – Inference & Interpretation
- [ ] Compute feature importance
- [ ] Run SHAP analysis for top model
- [ ] Interpret pricing impact vs competitor price
- [ ] Interpret campaign and availability effects
- [ ] Summarize business recommendations

### Phase 5 – Reporting & Presentation
- [ ] Create final figures for storytelling
- [ ] Build model comparison table
- [ ] Write paper/report (method, results, business aspect)
- [ ] Prepare on-site presentation slides
- [ ] Final review and cleanup of repository

---

## 📚 Source

Based on Data Mining Cup 2017 (adapted) — Andreas Reber, 05.02.2026

