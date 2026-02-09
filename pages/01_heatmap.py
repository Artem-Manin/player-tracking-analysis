import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Player Heatmap (GK toggle)")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("data/raw/session_001.csv")
df["EpochTime"] = pd.to_datetime(df["EpochTime"])

gk_df = pd.read_csv("figures/gk_intervals_3min.csv")
gk_df["start"] = pd.to_datetime(gk_df["start"])
gk_df["end"] = pd.to_datetime(gk_df["end"])

# ---------------------------
# Checkbox
# ---------------------------
show_gk = st.checkbox("Include goalkeeper intervals", value=True)

# ---------------------------
# Mark GK samples
# ---------------------------
df["is_gk"] = False

for _, row in gk_df[gk_df["gk"] == "YES"].iterrows():
    mask = (df["EpochTime"] >= row["start"]) & (df["EpochTime"] < row["end"])
    df.loc[mask, "is_gk"] = True

if not show_gk:
    df = df[df["is_gk"] == False]

# ---------------------------
# Heatmap bins (3 x 5)
# ---------------------------
lat = df["Latitude"].values
lon = df["Longitude"].values

lat_bins = np.linspace(lat.min(), lat.max(), 4)
lon_bins = np.linspace(lon.min(), lon.max(), 6)

heatmap, _, _ = np.histogram2d(
    lat,
    lon,
    bins=[lat_bins, lon_bins]
)

heatmap_pct = heatmap / heatmap.sum() * 100 if heatmap.sum() > 0 else heatmap

# ---------------------------
# Aspect ratio (meters)
# ---------------------------
lat_mid = np.mean(lat)
aspect_ratio = (
    (lon.max() - lon.min()) * np.cos(np.deg2rad(lat_mid))
) / (lat.max() - lat.min())

# ---------------------------
# Plot
# ---------------------------
fig, ax = plt.subplots(figsize=(4, 7))
ax.set_facecolor("#4C9F29")

img = ax.imshow(
    heatmap_pct.T[::-1],
    origin="lower",
    cmap="hot",
    alpha=0.65,
    extent=[0, 3, 0, 5],
    #aspect=aspect_ratio
)

# Grid lines
for x in range(4):
    ax.axvline(x, color="white", linewidth=1)
for y in range(6):
    ax.axhline(y, color="white", linewidth=1)

# Percent labels
data_for_labels = heatmap_pct.T[::-1]

for y in range(5):
    for x in range(3):
        val = data_for_labels[y, x]
        ax.text(
            x + 0.5, y + 0.5,
            f"{val:.1f}%",
            ha="center", va="center",
            color="white",
            fontsize=8, fontweight="bold"
        )

# Axis ticks
ax.set_xticks(np.arange(3) + 0.5)
ax.set_yticks(np.arange(5) + 0.5)

ax.set_xticklabels(["1", "2", "3"])
ax.set_yticklabels(["1", "2", "3", "4", "5"])

ax.set_xlabel("Field width (south → north)")
ax.set_ylabel("Field length (west → east)")

st.pyplot(fig, use_container_width=True)
