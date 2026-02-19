import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("data/raw/new_player_data_2026_02_06_174048 - new_player_data_2026_02_06_174048.csv")
df = df.dropna(subset=["Latitude", "Longitude"])

lat = df["Latitude"].values
lon = df["Longitude"].values

# ---------------------------
# Define bins
# ---------------------------
# 3 horizontal (south → north)
lat_bins = np.linspace(lat.min(), lat.max(), 4)

# 5 vertical (west → east)
lon_bins = np.linspace(lon.min(), lon.max(), 6)

# ---------------------------
# Histogram
# X = latitude  (horizontal, south → north)
# Y = longitude (vertical, west → east)
# ---------------------------
heatmap, _, _ = np.histogram2d(
    lat,
    lon,
    bins=[lat_bins, lon_bins]
)

heatmap_pct = heatmap / heatmap.sum() * 100

# ---------------------------
# Correct aspect ratio (meters!)
# ---------------------------
lat_mid = np.mean(lat)
meters_per_deg_lon = np.cos(np.deg2rad(lat_mid))

lat_range = lat.max() - lat.min()
lon_range = lon.max() - lon.min()

# Aspect = height / width
aspect_ratio = (lon_range * meters_per_deg_lon) / lat_range

# ---------------------------
# Plot
# ---------------------------
fig, ax = plt.subplots(figsize=(6, 10))
ax.set_facecolor("#4C9F29")  # green pitch

img = ax.imshow(
    heatmap_pct.T[::-1],     # transpose + flip → east at bottom
    origin="lower",
    cmap="hot",
    alpha=0.65,
    extent=[0, 3, 0, 5],
    #aspect=aspect_ratio
)

# Colorbar
cbar = fig.colorbar(img, ax=ax, pad=0.02)
cbar.set_label("% of time")

# ---------------------------
# Zone ticks (centered)
# ---------------------------
ax.set_xticks(np.arange(3) + 0.5)
ax.set_yticks(np.arange(5) + 0.5)

ax.set_xticklabels(["1", "2", "3"])
ax.set_yticklabels(["1", "2", "3", "4", "5"])

ax.set_xlabel("Field width (south → north)")
ax.set_ylabel("Field length (west → east)")

# ---------------------------
# Pitch grid lines
# ---------------------------
for x in range(4):
    ax.axvline(x, color="white", linewidth=1)

for y in range(6):
    ax.axhline(y, color="white", linewidth=1)

# ---------------------------
# Percent labels inside cells
# ---------------------------
data_for_labels = heatmap_pct.T[::-1]

for y in range(5):
    for x in range(3):
        val = data_for_labels[y, x]
        ax.text(
            x + 0.5,
            y + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

# ---------------------------
# Save
# ---------------------------
plt.tight_layout()
plt.savefig("figures/heatmap_pitch_3x5_realistic.png")
plt.close()

print("Saved: figures/heatmap_pitch_3x5_realistic.png")
