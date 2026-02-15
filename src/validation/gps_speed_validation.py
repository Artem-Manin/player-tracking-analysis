import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# VECTORISED HAVERSINE (FAST + SAFE)
# =========================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371000  # meters

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# =========================================================
# SPEED VALIDATION
# =========================================================

def validate_speed(df: pd.DataFrame):

    df = df.copy()

    # -------------------------
    # Parse timestamp robustly
    # -------------------------
    df["EpochTime"] = pd.to_datetime(df["EpochTime"], errors="coerce")
    df = df.dropna(subset=["EpochTime"])

    df = df.sort_values("EpochTime").reset_index(drop=True)

    df["t"] = df["EpochTime"].astype("int64") / 1e9
    df["dt"] = df["t"].diff()

    # -------------------------
    # Numeric coercion
    # -------------------------
    for col in ["Latitude", "Longitude", "Speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Latitude", "Longitude", "Speed", "dt"])
    df = df.reset_index(drop=True)

    # -------------------------
    # Vectorized distance
    # -------------------------
    lat1 = df["Latitude"].shift(1)
    lon1 = df["Longitude"].shift(1)
    lat2 = df["Latitude"]
    lon2 = df["Longitude"]

    df["gps_distance"] = haversine_vectorized(lat1, lon1, lat2, lon2)

    # first row has no previous point
    df.loc[0, "gps_distance"] = 0

    # -------------------------
    # GPS speed
    # -------------------------
    df["speed_gps"] = df["gps_distance"] / df["dt"]

    df.loc[(df["dt"] <= 0) | (df["dt"] > 2), "speed_gps"] = np.nan

    # -------------------------
    # Error metrics
    # -------------------------
    df["speed_error"] = df["Speed"] - df["speed_gps"]
    df["speed_flag"] = np.abs(df["speed_error"]) > 1.5

    report = {
        "rows_after_cleaning": int(len(df)),
        "mean_error": float(df["speed_error"].mean()),
        "median_error": float(df["speed_error"].median()),
        "rmse": float(np.sqrt((df["speed_error"]**2).mean())),
        "correlation": float(df[["Speed", "speed_gps"]].corr().iloc[0, 1]),
        "flagged_rows": int(df["speed_flag"].sum())
    }

    return df, report


# =========================================================
# FIGURES
# =========================================================

def save_figures(df, out_dir="figures"):

    os.makedirs(out_dir, exist_ok=True)

    # Time series
    plt.figure(figsize=(12,5))
    plt.plot(df["Speed"], label="Device speed", alpha=0.7)
    plt.plot(df["speed_gps"], label="GPS speed", alpha=0.7)
    plt.legend()
    plt.title("Speed comparison")
    plt.xlabel("Sample")
    plt.ylabel("Speed (m/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "speed_timeseries.png"))
    plt.close()

    # Scatter
    plt.figure(figsize=(6,6))
    plt.scatter(df["Speed"], df["speed_gps"], s=6, alpha=0.5)
    maxv = np.nanmax([df["Speed"].max(), df["speed_gps"].max()])
    plt.plot([0, maxv], [0, maxv], "--")
    plt.xlabel("Device speed (m/s)")
    plt.ylabel("GPS speed (m/s)")
    plt.title("Identity plot")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "speed_scatter.png"))
    plt.close()

    # Error histogram
    plt.figure(figsize=(8,5))
    plt.hist(df["speed_error"].dropna(), bins=60)
    plt.xlabel("Speed error (m/s)")
    plt.ylabel("Count")
    plt.title("Speed error distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "speed_error_hist.png"))
    plt.close()


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":

    DATA_PATH = (
        "data/raw/"
        "new_player_data_2026_02_06_174048 - "
        "new_player_data_2026_02_06_174048.csv"
    )

    OUTPUT_PATH = "data/raw/gps_validated.csv"

    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)

    print("Running GPS speed validation...")
    df_validated, report = validate_speed(df)

    print("\nGPS SPEED VALIDATION REPORT")
    for k, v in report.items():
        print(f"{k}: {v}")

    print("\nSaving CSV...")
    df_validated.to_csv(OUTPUT_PATH, index=False)

    print("Saving figures...")
    save_figures(df_validated)

    print("\nDone.")
