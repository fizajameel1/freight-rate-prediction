import joblib
import pandas as pd

import features as F


def city_coords(train, val):
    """One coordinate pair per city (taken from the provided data)."""
    both = pd.concat([train, val])
    pick = both.drop_duplicates("pickup").set_index("pickup")[["pickup_lat", "pickup_lon"]]
    deliv = both.drop_duplicates("delivery").set_index("delivery")[["delivery_lat", "delivery_lon"]]
    return pick, deliv


def main():
    bundle = joblib.load("models/freight_model.joblib")
    model, med = bundle["model"], bundle["weight_med"]
    train = pd.read_csv("data/train_test.csv")
    val = pd.read_csv("data/validation.csv")

    # 1) Final predictions for the 12,000 validation loads
    template = pd.read_csv("data/validation_predictions_template.csv")
    feats = F.add_features(val, med)
    feats["predicted_rate"] = model.predict(feats[F.FEATURES]) * feats["distance"]
    out = template[["load_id"]].merge(feats[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and len(out) == 12000
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv("validation_predictions.csv", index=False)
    print("Wrote validation_predictions.csv:", out.shape)

    # 2) Fixed December scenario (only the date changes)
    dec = pd.read_csv("data/december_chart_inputs.csv")
    pick, deliv = city_coords(train, val)
    d = dec.copy()
    d[["pickup_lat", "pickup_lon"]] = pick.loc[d["pickup"]].to_numpy()
    d[["delivery_lat", "delivery_lon"]] = deliv.loc[d["delivery"]].to_numpy()
    d = F.add_features(d, med)
    dec["predicted_rate"] = (model.predict(d[F.FEATURES]) * d["distance"]).round(2)
    dec.to_csv("data/december_chart_inputs.csv", index=False)
    print("Filled data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()