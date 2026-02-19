import pandas as pd
import numpy as np

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("data/raw/new_player_data_2026_02_06_174048 - new_player_data_2026_02_06_174048.csv")
df = df.dropna(subset=["EpochTime", "Latitude", "Longitude", "Speed"])

df["EpochTime"] = pd.to_datetime(df["EpochTime"])
df = df.sort_values("EpochTime")

# ---------------------------
# Define bins (same as heatmap)
# ---------------------------
lat = df["Latitude"].values
lon = df["Longitude"].values

lat_bins = np.linspace(lat.min(), lat.max(), 4)   # 3 horizontal
lon_bins = np.linspace(lon.min(), lon.max(), 6)   # 5 vertical

# Assign zones
df["lat_zone"] = np.digitize(df["Latitude"], lat_bins) - 1
df["lon_zone"] = np.digitize(df["Longitude"], lon_bins) - 1

# Clip just in case
df["lat_zone"] = df["lat_zone"].clip(0, 2)
df["lon_zone"] = df["lon_zone"].clip(0, 4)

# GK target zone
GK_LAT_ZONE = 1   # middle (south-north)
GK_LON_ZONE = 4   # bottom (west-east)

# ---------------------------
# 3-minute windows
# ---------------------------
WINDOW = pd.Timedelta(minutes=3)

results = []

start_times = pd.date_range(
    df["EpochTime"].min(),
    df["EpochTime"].max(),
    freq=WINDOW
)

for start in start_times:
    end = start + WINDOW
    w = df[(df["EpochTime"] >= start) & (df["EpochTime"] < end)]

    if len(w) < 10:
        continue

    # --- Features ---
    in_gk_zone = (
        (w["lat_zone"] == GK_LAT_ZONE) &
        (w["lon_zone"] == GK_LON_ZONE)
    )

    pct_in_gk_zone = in_gk_zone.mean() * 100

    duration_sec = (w["EpochTime"].iloc[-1] - w["EpochTime"].iloc[0]).total_seconds()

    total_distance = w["Speed"].sum() * 0.5  # approx, since ~0.5s sampling
    max_speed = w["Speed"].max()

    # --- Decision ---
    gk = (
        pct_in_gk_zone >= 60 and
        total_distance < 300 and
        max_speed < 5
    )

    results.append({
        "start": start,
        "end": end,
        "gk": "YES" if gk else "NO",
        "pct_in_gk_zone": round(pct_in_gk_zone, 1),
        "distance_m": round(total_distance, 1),
        "max_speed": round(max_speed, 2)
    })

# ---------------------------
# Result table
# ---------------------------
gk_df = pd.DataFrame(results)
gk_df.to_csv("figures/gk_intervals_3min.csv", index=False)

print(gk_df)
print("\nSaved: figures/gk_intervals_3min.csv")
