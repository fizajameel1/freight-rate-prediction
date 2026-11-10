from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

import features as F

PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, random_state=42, verbose=-1,
)


def fit_model(df):
    """Train on rate per mile (rpm); distance is applied afterwards."""
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(df[F.FEATURES], df["rpm"])
    return model


def predict_rate(model, df):
    return model.predict(df[F.FEATURES]) * df["distance"]


def report(name, pred, actual):
    err = (pred - actual).abs()
    print(f"{name}: MAE ${err.mean():.1f} | MAPE {(err / actual).mean() * 100:.2f}%")


def main():
    raw = pd.read_csv("data/train_test.csv")
    dev = F.drop_bad_labels(raw)
    print(f"Training rows after removing bad labels: {len(dev)} (removed {len(raw) - len(dev)})")

    # Time-based validation: fit on Jan-Aug, hold out Sep-Oct
    month = pd.to_datetime(dev["date"]).dt.month
    fit_df, hold_df = dev[month <= 8], dev[month >= 9]
    
    med = F.weight_medians(fit_df)
    fit_df = F.add_features(fit_df, med)
    hold_df = F.add_features(hold_df, med)
    print(f"Fit rows: {len(fit_df)} | Holdout rows: {len(hold_df)}")

    baseline = fit_df.groupby("equipment")["rpm"].median()
    report("Baseline (median rate/mile by equipment)",
           hold_df["distance"] * hold_df["equipment"].map(baseline), hold_df["posted_rate"])
    model = fit_model(fit_df)
    report("LightGBM on Sep-Oct holdout            ",
           predict_rate(model, hold_df), hold_df["posted_rate"])

    # Final model: retrain on all cleaned development data
    med = F.weight_medians(dev)
    final = fit_model(F.add_features(dev, med))
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "weight_med": med}, "models/freight_model.joblib")
    print("Saved models/freight_model.joblib")


if __name__ == "__main__":
    main()