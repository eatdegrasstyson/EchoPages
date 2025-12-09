import pandas as pd
import matplotlib.pyplot as plt

# Path to your dataset
csv_path = "SpotifyMetaToGems_Final.csv"

# Load CSV
df = pd.read_csv(csv_path)

# List of the 9 GEMs emotion columns
gems_cols = [
    "Wonder",
    "Transcendence",
    "Tenderness",
    "Nostalgia",
    "Peacefulness",
    "Power",
    "Joy",
    "Tension",
    "Sadness"
]

# Calculate total sum for each emotion
emotion_totals = df[gems_cols].sum()

# Convert to percentages
emotion_percent = (emotion_totals / emotion_totals.sum()) * 100

# Plot
plt.figure(figsize=(10, 5))
emotion_percent.plot(kind="bar")
plt.title("GEMs Emotion Distribution (%)")
plt.ylabel("Percentage (%)")
plt.xlabel("Emotion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
