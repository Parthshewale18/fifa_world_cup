# FIFA World Cup 2026 Match Outcome Predictor

An end-to-end machine learning project that predicts Win / Draw / Loss outcomes for FIFA World Cup 2026 matches, built and tested live during the tournament itself.

## Overview

This project collects historical football data (1872–present), FIFA rankings, and 2026-specific team features, engineers predictive features (team form, head-to-head history, rank differential), and trains a classifier to predict match outcomes. Predictions were generated for real, unplayed matches — including the 2026 semifinals — and checked against actual results as the tournament unfolded.

## Data Sources

| Dataset | Source | Role |
|---|---|---|
| International football results (1872–present) | [Kaggle: martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Main training data — match-level results |
| FIFA World Ranking history | [Kaggle: cashncarry](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Team strength feature at time of match |
| FIFA World Cup Team Dataset (2002–2026) | [Kaggle: harrachimustapha](https://www.kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset) | Bonus team-level features (squad value, host status, tournament history) |

Only matches from the **last 25 years** are used for training — older matches don't reflect the modern game closely enough to be useful signal.

## Project Structure

```
worldcup2026-predictor/
├── data/
│   ├── raw/                  # downloaded/collected source CSVs
│   └── processed/            # cleaned, merged, model-ready datasets
├── notebooks/
│   ├── clean_and_merge.ipynb     # loads, cleans, merges all raw sources
│   ├── eda.ipynb                 # exploratory analysis, missing values, outliers
│   ├── feature_engineering.ipynb # builds recent form, head-to-head, rest days; encodes target
│   └── model.ipynb               # trains + compares baseline vs XGBoost, evaluates, saves model
├── src/
│   └── data/
│       └── data_inspection.py   # prints columns/shape for every raw CSV
│       └── predict.py   # generates predictions for unplayed matches
├── models/                   # trained model + feature column list (not committed — see below)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Download the three datasets linked above into `data/raw/historical_matches/`, `data/raw/fifa_rankings/`, and `data/raw/wc_team_features_2026/` respectively.

## Pipeline

Run the notebooks in order:

1. **`clean_and_merge.ipynb`** — loads raw CSVs, standardizes team names, filters to last 25 years, attaches each team's FIFA ranking at time of match (using `merge_asof` to avoid future leakage), outputs `matches_merged_cleaned.csv`
2. **`eda.ipynb`** — checks missing values, class balance, home advantage, outlier scorelines
3. **`feature_engineering.ipynb`** — builds recent form (rolling last-5-match win rate and goals), head-to-head win rate, rest days, buckets and encodes tournament type, encodes the target — outputs `model_features.csv`
4. **`model.ipynb`** — chronological train/test split, naive baseline vs Logistic Regression vs XGBoost, evaluation, saves trained model to `models/`

## Model & Results

**Target:** Home win / Draw / Away win (3-class classification)

**Features used:** FIFA rank and points (home/away), rank differential, recent form (win rate, goals for/against), **head-to-head win rate**, tournament type (World Cup / qualifier / continental championship / friendly), neutral venue flag.

| Model | Accuracy |
|---|---|
| Naive baseline (`rank_diff > 0` → home win) | 43% |
| Logistic Regression | 48% |
| XGBoost (without head-to-head feature) | 50.7% |
| **XGBoost + head-to-head win rate** | **54.5%** |

XGBoost log-loss: **0.974** (improved from 1.038)

```
              precision    recall  f1-score   support
Away             0.49      0.48      0.49       378
Draw             0.43      0.02      0.04       280
Home             0.57      0.82      0.67       622
```

**Update:** adding `head-to-head win rate` — the historical win rate between the two specific teams facing off — improved accuracy from 50.7% to 54.5% and reduced log-loss from 1.038 to 0.974. This was the single most impactful feature addition in the project so far, confirming that a team's history against a *specific* opponent carries real predictive signal beyond just general ranking and form.

**Note on accuracy:** 50-55% is broadly in line with published football outcome prediction research — even professional models rarely clear 55-60% on 3-way classification. The harder challenge remains the **Draw class** (recall of just 0.02), which is notoriously difficult to call from pre-match stats alone; this is a known, active limitation being addressed in ongoing iterations (see Next Steps).

Feature importance confirmed the model is learning sensible patterns: `h2h_win_rate` ,rank_diff, rank/points, recent form, and now head-to-head history are the strongest predictors — no evidence of data leakage or spurious correlations.

## Predicting Upcoming Matches

`src/data/predict.py` loads the trained model and generates outcome probabilities for any two teams, using each team's most recent known rank/form/points:

```bash
python src/data/predict.py
```

Example output:
```
France vs Spain (2026-07-15)
  France: 41.0%
  Draw: 18.4%
  Spain: 40.5%
  --> Predicted: France
```
This came in very close to the real result.

```
England vs Argentina (2026-07-16)
  England: 31.3%
  Draw: 26.7%
  Argentina: 41.9%
  --> Predicted: Argentina
```

## Next Steps

- Add `rest days` (days since each team's last match) back into the trained feature set (currently engineered but not yet included in the production model)
- Address class imbalance for the Draw category using sample weighting
- Backtest systematically against all 2026 World Cup matches played so far
- Build a simple Streamlit dashboard for interactive predictions

## Tech Stack

Python, pandas, scikit-learn, XGBoost, seaborn/matplotlib, Jupyter

## Data License

Datasets used are publicly available on Kaggle under their respective licenses. This project is for educational/portfolio purposes.