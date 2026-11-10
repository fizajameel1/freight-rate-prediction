# Freight Rate Prediction

Predicts the posted rate of freight loads. Validated with a time-based split;
final predictions are for the 12,000 loads in `data/validation.csv` (Nov-Dec 2025).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
Place the provided CSV files (train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv) in a `data/` folder before running.
```

## Run (from the project root)

```bash
python src/train.py      # cleans data, runs Sep-Oct holdout validation, saves the model
python src/predict.py    # writes validation_predictions.csv and fills the December chart file
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Approach

- **Data cleaning:** 340 training rows with corrupted rates (rate per mile more than 2x the
  typical value for similar distances) are removed. Negative weights (sign errors) are made
  positive. Missing weights are filled with the median for the equipment type.
- **Validation:** train on Jan-Aug 2025, test on Sep-Oct 2025 (the real task predicts the
  two months after the training data). A second stress test holds out the weeks where
  `quote_signal` is uninformative.
- **Model:** LightGBM predicting rate per mile, multiplied by distance. Features: distance,
  equipment, weight, pickup/delivery coordinates, day of week.
- **Deliberately excluded:** `quote_signal` (its relationship with the rate flips sign between
  periods, and validation looks like the unreliable case) and `market_index` (it hurt accuracy
  in both tests).
- **Results on the Sep-Oct holdout:** baseline MAE $184.8 (10.0% MAPE) vs model MAE $58.4 (4.05% MAPE).

## Files

- `src/features.py` cleaning and feature engineering
- `src/train.py` validation and model training
- `src/predict.py` predictions and December chart inputs
- `notebooks/01_eda.ipynb` data exploration
- `scorer_results/candidate_december.png` December chart from `score.py`