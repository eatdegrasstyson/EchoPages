import os
import pandas as pd
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
muvi_file = os.path.join(project_root, "DataSets", "MuVi", "media_data.csv")
output_file = os.path.join(project_root, "DataSets", "MuVi", "parsed_songs.csv")

df = pd.read_csv(muvi_file)

#Keep only music or video entries
df = df[df["media_modality"].isin(["music", "video"])]

#Drop irrelevant columns
df = df.drop(columns=["media_modality", "seen_previously", "overlay_type"])

#Custom Soft mapping of GEMS-28 -> GEMS-9
soft_gems9_mapping = {
    "Wonder": ["filled_with_wonder", "allured", "fascinated"],
    "Transcendence": ["feeling_of_transcendence", "serene", "overwhelmed"],
    "Tenderness": ["tender", "affectionate", "mellow"],
    "Nostalgia": ["nostalgic", "sentimental", "dreamy"],
    "Peacefulness": ["calm", "soothed"],
    "Power": ["strong", "energetic", "triumphant"],
    "Joy": ["animated", "bouncy", "joyful"],
    "Tension": ["tense", "agitated", "nervous"],
    "Sadness": ["sad", "tearful", "blue"]
}

gems9_labels = list(soft_gems9_mapping.keys())

#Initialize dictionary for aggregated song data
song_vectors = defaultdict(lambda: defaultdict(float))
song_counts = defaultdict(int)

#Process each row (participant rating)
for _, row in df.iterrows():
    media_id = row["media_id"]
    song_counts[media_id] += 1
    
    for gems9_label, original_emotions in soft_gems9_mapping.items():
        contrib = 0.0
        for emotion in original_emotions:
            if row.get(emotion, 0) > 0:
                contrib += 1.0 / len(original_emotions)
        song_vectors[media_id][gems9_label] += contrib

#Compute weighted average per song and normalize
final_data = []
for media_id, vec in song_vectors.items():
    count = song_counts[media_id]
    averaged_vec = {label: vec[label] / count for label in gems9_labels}

    total = sum(averaged_vec.values())
    if total > 0:
        normalized_vec = {label: val / total for label, val in averaged_vec.items()}
    else:
        normalized_vec = averaged_vec

    normalized_vec["media_id"] = media_id
    final_data.append(normalized_vec)

#Convert to DataFrame and save
final_df = pd.DataFrame(final_data)
final_df = final_df[["media_id"] + gems9_labels]
final_df.to_csv(output_file, index=False)

#Print normalized totals across dataset
totals = final_df[gems9_labels].sum()
print("Normalized GEMS-9 totals across dataset (sum of each column):")
for label, total in totals.items():
    print(f"- {label}: {total:.2f}")

print(f"\nNumber of unique songs classified: {len(final_df)}")
print(f"\nParsed and weighted MuVi songs saved to {output_file}")
