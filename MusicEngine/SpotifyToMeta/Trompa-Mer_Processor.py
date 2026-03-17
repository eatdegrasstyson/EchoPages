import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

trompa_file = os.path.join(project_root, "DataSets", "Trompa-Mer", "summary_anno.csv")
output_dir = os.path.join(project_root, "DataSets", "Trompa-Mer", "ParsedData")
output_file = os.path.join(output_dir, "filtered_trompa_mer.csv")
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(trompa_file):
    raise FileNotFoundError(f"Expected file not found: {trompa_file}")

df = pd.read_csv(trompa_file, sep="\t")

columns_to_keep = [
    "track_id", "composer_name", "artist_name", "track_name",
    "tenderness", "sadness", "joy", "transcendence", "power",
    "tension", "fear", "peace", "bitterness", "anger", "surprise"
]
existing_columns = [col for col in columns_to_keep if col in df.columns]
df_filtered = df[existing_columns]

df_filtered = df_filtered.dropna(subset=["composer_name", "artist_name", "track_name"])
df_filtered = df_filtered[~(
    (df_filtered["composer_name"].str.lower() == "not found") &
    (df_filtered["artist_name"].str.lower() == "not found") &
    (df_filtered["track_name"].str.lower() == "not found")
)]

mood_columns = [c for c in existing_columns if c not in ["track_id", "composer_name", "artist_name", "track_name"]]
df_filtered = df_filtered.dropna(subset=mood_columns, how="all")

df_filtered.to_csv(output_file, index=False)

print(f"Filtered TROMPA-MER dataset saved to: {output_file}")
print(f"   Total songs: {len(df_filtered)}")
print(f"   Columns kept: {', '.join(existing_columns)}")
