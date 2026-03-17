import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("Emotify/data.csv")
df.columns = df.columns.str.strip()

rename_map = {
    "amazement": "Wonder",
    "solemnity": "Transcendence",
    "tenderness": "Tenderness",
    "nostalgia": "Nostalgia",
    "calmness": "Peacefulness",
    "power": "Power",
    "joyful_activation": "Joy",
    "tension": "Tension",
    "sadness": "Sadness"
}

df = df.rename(columns=rename_map)

drop_cols = ["liked", "disliked", "age", "gender", "mother tongue", "mood"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

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

grouped = df.groupby(["track id", "genre"], as_index=False)[gems_cols].mean()

def normalize_row(row):
    total = row.sum()
    return row / total if total != 0 else row

grouped[gems_cols] = grouped[gems_cols].apply(normalize_row, axis=1)

final_matrix = []

for _, row in grouped.iterrows():
    tid = row["track id"]
    genre = row["genre"]
    gems_vector = row[gems_cols].values.tolist()

    final_matrix.append([tid, genre] + gems_vector)

final_matrix = np.array(final_matrix, dtype=object)

out_cols = ["track_id", "genre"] + gems_cols
df_out = pd.DataFrame(final_matrix, columns=out_cols)
df_out.to_csv("testset.csv", index=False)
