import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_FILE = "data/raw/new_player_data_2026_02_06_174048 - new_player_data_2026_02_06_174048.csv"
WINDOW_MINUTES = 3

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["EpochTime", "Latitude", "Longitude", "Speed"])

df["EpochTime"] = pd.to_datetime(df["EpochTime"])
df = df.sort_values("EpochTime")

# -------------------------------------------------
# Spatial bins (same as heatmap)
# -------------------------------------------------
lat_bins = np.linspace(df["Latitude"].min(), df["Latitude"].max(), 4)   # 3 zones
lon_bins = np.linspace(df["Longitude"].min(), df["Longitude"].max(), 6) # 5 zones

df["lat_zone"] = np.digitize(df["Latitude"], lat_bins) - 1
df["lon_zone"] = np.digitize(df["Longitude"], lon_bins) - 1

df["lat_zone"] = df["lat_zone"].clip(0, 2)
df["lon_zone"] = df["lon_zone"].clip(0, 4)

GK_LAT_ZONE = 1   # central
GK_LON_ZONE = 4   # defensive bottom

# -------------------------------------------------
# Build 3-min windows
# -------------------------------------------------
WINDOW = pd.Timedelta(minutes=WINDOW_MINUTES)

rows = []
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

    rows.append({
        "start": start,
        "pct_def_third": (w["lon_zone"] == GK_LON_ZONE).mean(),
        "pct_central": (w["lat_zone"] == GK_LAT_ZONE).mean(),
        "mean_speed": w["Speed"].mean(),
        "max_speed": w["Speed"].max(),
        "total_distance": w["Speed"].sum() * 0.5,
        "pct_low_speed": (w["Speed"] < 1.0).mean(),
        "accel_std": w["Speed"].diff().std()
    })

feature_df = pd.DataFrame(rows)

# -------------------------------------------------
# Feature matrix
# -------------------------------------------------
X = feature_df[
    [
        "pct_def_third",
        "pct_central",
        "mean_speed",
        "max_speed",
        "total_distance",
        "pct_low_speed",
        "accel_std"
    ]
].values

# -------------------------------------------------
# Standardize
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# KMeans
# -------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
feature_df["cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------------------------
# Determine GK cluster (lowest mean speed)
# -------------------------------------------------
centroids = feature_df.groupby("cluster").mean()
gk_cluster = centroids["mean_speed"].idxmin()

feature_df["role_raw"] = feature_df["cluster"].apply(
    lambda c: "GK" if c == gk_cluster else "Field"
)

# -------------------------------------------------
# Temporal smoothing
# -------------------------------------------------
roles = feature_df["role_raw"].tolist()
smoothed = roles.copy()

for i in range(1, len(roles) - 1):
    if roles[i-1] == roles[i+1] and roles[i] != roles[i-1]:
        smoothed[i] = roles[i-1]

feature_df["role"] = smoothed
N_TRIM = 2
feature_df = feature_df.iloc[N_TRIM:-N_TRIM]

# -------------------------------------------------
# Print results
# -------------------------------------------------
print("\nCluster centroids:")
print(centroids)

print("\nGK / Field per window:")
print(feature_df[["start", "role"]])

# -------------------------------------------------
# Save CSV
# -------------------------------------------------
feature_df[["start", "role"]].to_csv(
    "figures/gk_intervals_clustered.csv",
    index=False
)

# -------------------------------------------------
# Visualization
# -------------------------------------------------
plt.figure(figsize=(6,4))

for c in feature_df["cluster"].unique():
    subset = feature_df[feature_df["cluster"] == c]
    plt.scatter(
        subset["mean_speed"],
        subset["total_distance"],
        label=f"Cluster {c}",
        alpha=0.7
    )

plt.xlabel("Mean speed")
plt.ylabel("Total distance")
plt.title("GK vs Field clustering")
plt.legend()
plt.tight_layout()
plt.savefig("figures/gk_cluster_scatter.png")
plt.close()

print("\nSaved: figures/gk_intervals_clustered.csv")
print("Saved: figures/gk_cluster_scatter.png")
