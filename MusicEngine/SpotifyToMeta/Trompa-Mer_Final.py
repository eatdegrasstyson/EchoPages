import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

input_file = os.path.join(project_root, "DataSets", "Trompa-Mer", "ParsedData", "trompa_mer_with_spotify_ids.csv")
output_file = os.path.join(project_root, "DataSets", "Trompa-Mer", "ParsedData", "trompa_mer_final.csv")

#Error handling if input file doesnt exist:
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Expected file not found: {input_file}")

df = pd.read_csv(input_file)

gems11 = ["tenderness","sadness","joy","transcendence","power","tension","fear","peace","bitterness","anger","surprise"]

df = df[["track_id"] + gems11]
df = df[df["track_id"].notna() & (df["track_id"] != "NA")]
df = df[~(df[gems11]==0).all(axis=1)]

print("GEMS11 totals across dataset:")
print(df[gems11].sum())

gems11_to_9 = {
    "tenderness": "Tenderness",
    "sadness": "Sadness",
    "joy": "Joy",
    "transcendence": "Transcendence",
    "power": "Power",
    "tension": "Tension",
    "fear": "Tension",
    "peace": "Peacefulness",
    "bitterness": "Sadness",
    "anger": "Tension",
    "surprise": "Wonder"
}

gems9_labels = ["Wonder","Transcendence","Tenderness","Nostalgia","Peacefulness","Power","Joy","Tension","Sadness"]

rows = []
for _, row in df.iterrows():
    if not row["track_id"] or row["track_id"]=="NA":
        continue
    new_row = {"track_id": row["track_id"]}
    for label in gems9_labels:
        new_row[label] = 0.0

    for gem11_label, value in row[gems11].items():
        mapped_label = gems11_to_9.get(gem11_label)
        if mapped_label:
            new_row[mapped_label] += value

    total = sum(new_row[label] for label in gems9_labels)
    if total > 0:
        for label in gems9_labels:
            new_row[label] /= total

    rows.append(new_row)

df_final = pd.DataFrame(rows)
df_final = df_final[["track_id"] + gems9_labels]
df_final.to_csv(output_file, index=False)

print("GEMS9 totals across dataset:")
print(df_final[gems9_labels].sum())

print(f"GEMS9 data saved to {output_file}, total songs: {len(df_final)}")
