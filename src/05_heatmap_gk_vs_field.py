import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_FILE = "data/raw/new_player_data_2026_02_06_174048 - new_player_data_2026_02_06_174048.csv"
GK_FILE = "figures/gk_intervals_clustered.csv"
OUT_FILE = "figures/heatmap_gk_vs_field.png"

WINDOW = pd.Timedelta(minutes=3)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(DATA_FILE)
df["EpochTime"] = pd.to_datetime(df["EpochTime"])

gk_df = pd.read_csv(GK_FILE)
gk_df["start"] = pd.to_datetime(gk_df["start"])

# --------------------------------------------------
# Assign role to each GPS sample
# --------------------------------------------------
df["role"] = "Field"

for _, row in gk_df.iterrows():
    start = row["start"]
    end = start + WINDOW
    mask = (df["EpochTime"] >= start) & (df["EpochTime"] < end)
    df.loc[mask, "role"] = row["role"]

# --------------------------------------------------
# Create bins (ROWS = latitude zones = 3)
#              COLS = longitude zones = 5
# --------------------------------------------------
lat_bins = np.linspace(df["Latitude"].min(), df["Latitude"].max(), 4)
lon_bins = np.linspace(df["Longitude"].min(), df["Longitude"].max(), 6)

# --------------------------------------------------
# Heatmap builder (percent)
# --------------------------------------------------
def build_heatmap(sub):
    heat, _, _ = np.histogram2d(
        sub["Latitude"],
        sub["Longitude"],
        bins=[lat_bins, lon_bins]
    )

    total = heat.sum()
    if total == 0:
        return np.zeros((3,5))

    return (heat / total) * 100


heat_gk = build_heatmap(df[df["role"] == "GK"])
heat_field = build_heatmap(df[df["role"] == "Field"])

print("GK heatmap:\n", heat_gk)
print("Field heatmap:\n", heat_field)

# --------------------------------------------------
# Plot PURE MATRIX
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

titles = ["Goalkeeper Heatmap", "Field Player Heatmap"]
heats = [heat_gk, heat_field]

for ax, heat, title in zip(axes, heats, titles):

    img = ax.imshow(
        heat,
        cmap="plasma",
        vmin=0,
        vmax=100
    )

    # Grid
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(3))
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False)

    # Labels
    for i in range(3):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{heat[i,j]:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold"
            )

    ax.set_title(title)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150)
plt.close()

print(f"Saved: {OUT_FILE}")
