import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load
df = pd.read_csv("data/raw/session_001.csv")
df = df.dropna(subset=["Latitude", "Longitude", "EpochTime"])
df["EpochTime"] = pd.to_datetime(df["EpochTime"])

lat = df["Latitude"].values
lon = df["Longitude"].values
time = df["EpochTime"]

# Bins
lat_bins = np.linspace(lat.min(), lat.max(), 4)   # 3 wide
lon_bins = np.linspace(lon.min(), lon.max(), 6)   # 5 long

window = pd.Timedelta(seconds=60)
times = pd.date_range(time.min(), time.max(), freq="30s")

fig, ax = plt.subplots(figsize=(5, 10))
ax.set_facecolor("#4C9F29")

img = ax.imshow(np.zeros((5,3)), origin="lower", cmap="hot",
                extent=[0,3,0,5], alpha=0.7)

def update(t):
    mask = (time >= t) & (time < t + window)
    h,_,_ = np.histogram2d(
        df.loc[mask,"Latitude"],
        df.loc[mask,"Longitude"],
        bins=[lat_bins, lon_bins]
    )
    if h.sum() > 0:
        h = h / h.sum() * 100
    img.set_data(h.T[::-1])
    ax.set_title(f"{t.strftime('%H:%M:%S')} – { (t+window).strftime('%H:%M:%S') }")
    return img,

ani = FuncAnimation(fig, update, frames=times, interval=400)
ani.save("figures/heatmap_animation.gif")
