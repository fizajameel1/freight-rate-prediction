import pandas as pd

BUCKETS = [0, 250, 500, 1000, 2000, 4000]
EQUIPMENT = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
COORDS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
FEATURES = ["distance", "eq", "weight"] + COORDS + ["dow"]


def drop_bad_labels(df, threshold=2.0):
    """Remove training rows whose rate per mile is more than `threshold` times
    the typical rate per mile for loads of similar distance (corrupted labels)."""
    df = df.copy()
    df["rpm"] = df["posted_rate"] / df["distance"]
    bucket = pd.cut(df["distance"], BUCKETS)
    typical = df.groupby(bucket, observed=True)["rpm"].transform("median")
    return df[df["rpm"] / typical <= threshold].copy()


def weight_medians(df):
    """Median weight per equipment type (used to fill missing weights)."""
    return df["weight"].abs().groupby(df["equipment"]).median()


def add_features(df, weight_med):
    """Fix sign errors, fill missing weights, and build the model features."""
    df = df.copy()
    df["weight"] = df["weight"].abs()
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_med))
    df["eq"] = df["equipment"].map(EQUIPMENT)
    df["dow"] = pd.to_datetime(df["date"]).dt.day_of_week
    return df