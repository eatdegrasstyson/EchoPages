import pandas as pd
import matplotlib.pyplot as plt

# Paths
gems_csv = "DataSets/SpotifyMetaToGems_Final.csv"
meta_csv = "DataSets/SpotifyIDMappings.csv"

# Load data
gems_df = pd.read_csv(gems_csv)
meta_df = pd.read_csv(meta_csv)

# Keep only rows with a valid mp3 path
meta_df = meta_df[meta_df["mp3"].notna() & (meta_df["mp3"] != "")]

# Merge on spotifyid (adjust if your key is different)
df = pd.merge(meta_df, gems_df, on="spotifyid")

# GEMs columns
gems_cols = [
    "Wonder", "Transcendence", "Tenderness",
    "Nostalgia", "Peacefulness", "Power",
    "Joy", "Tension", "Sadness"
]

# --- Weighted aggregation ---
# Multiply each emotion by n_frames
weighted = df[gems_cols].multiply(df["n_frames"], axis=0)

# Sum weighted values
emotion_totals = weighted.sum()

# Normalize to percentage
emotion_percent = (emotion_totals / emotion_totals.sum()) * 100

# Plot
plt.figure(figsize=(10, 5))
emotion_percent.plot(kind="bar")
plt.title("Length-Weighted GEMs Emotion Distribution (%)")
plt.ylabel("Percentage (%)")
plt.xlabel("Emotion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()