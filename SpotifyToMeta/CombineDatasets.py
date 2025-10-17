import os
import pandas as pd
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

input_file1 = os.path.join(project_root, "DataSets", "Trompa-Mer", "ParsedData", "trompa_mer_final.csv")
input_file2 = os.path.join(project_root, "DataSets", "MuVi", "ParsedData", "final.csv")
input_file3 = os.path.join(project_root, "DataSets", "Music-Mouv", "ParsedData", "parsed_songs.json")
output_file = os.path.join(project_root, "DataSets", "SpotifyMetaToGems_Final.csv")

#Parse Music-Mouv
with open(input_file3, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for track_id, track_info in data.items():
    gems_vector = track_info.get("gems_vector", {})
    row = {"spotifyid": track_id}
    row.update(gems_vector)
    rows.append(row)

df = pd.DataFrame(rows)

desired_order = ["Wonder","Transcendence","Tenderness","Nostalgia","Peacefulness","Power","Joy","Tension","Sadness"]
df = df[["spotifyid"] + desired_order]

#Parse MuVi
df2 = pd.read_csv(input_file2)

#Parse Troma-Mer
df3 = pd.read_csv(input_file1)

df2 = df2.rename(columns={"media_id": "spotifyid"})
df3 = df3.rename(columns={"track_id": "spotifyid"})
df2 = df2[["spotifyid"] + desired_order]
df3 = df3[["spotifyid"] + desired_order]

df.insert(1, "DataSet", "Music-Mouv")
df2.insert(1, "DataSet", "MuVi")
df3.insert(1, "DataSet", "Trompa-Mer")

combined_df = pd.concat([df, df2, df3], ignore_index=True)

duplicates = combined_df[combined_df.duplicated(subset="spotifyid", keep=False)]
duplicate_ids = duplicates["spotifyid"].unique()
print(f"Duplicate track IDs ({len(duplicate_ids)}): {duplicate_ids}")

# combined_df = combined_df.groupby("spotifyid", as_index=False).agg({**{col: "sum" for col in desired_order}, "DataSet": lambda x: " & ".join(sorted(x.unique()))})
# cols = ["spotifyid", "DataSet"] + desired_order
# combined_df = combined_df[cols]
# combined_df[desired_order] = combined_df[desired_order].div(combined_df[desired_order].sum(axis=1), axis=0)

for idx, row in combined_df.iterrows():
    total = row[desired_order].sum()
    assert abs(total - 1.0) < 1e-6, f"Row {idx} does not sum to 1, sum={total}"

combined_df.to_csv(output_file, index=False)
print(f"Combined dataset saved to {output_file}, total tracks: {len(combined_df)}")
