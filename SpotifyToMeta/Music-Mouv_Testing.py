import json
import os
from collections import defaultdict, Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
parsed_file = os.path.join(project_root, "DataSets", "Music-Mouv", "ParsedData", "participants_tests_simplified.json")
output_file = os.path.join(project_root, "DataSets", "Music-Mouv", "ParsedData", "parsed_songs.json")

# Define canonical GEMS-9 categories
GEMS_9 = ["Joy", "Nostalgia", "Peacefulness", "Sadness",
          "Tenderness", "Tension", "Transcendence", "Wonder", "Power"]

#Load simplified parsed data
with open(parsed_file, "r", encoding="utf-8") as f:
    participants_tests = json.load(f)

#Track -> list of labels
track_labels = defaultdict(list)
track_metadata = {}

for participant, tests in participants_tests.items():
    for test in tests:
        gem_label = test.get("standardized_esthetic_emotion")
        track_id = test["music_data"]["spotify_data"]["track_id"]

        #Store label for weighting later
        if gem_label in GEMS_9:
            track_labels[track_id].append(gem_label)

            #Save track metadata (just once per track_id)
            if track_id not in track_metadata:
                track_metadata[track_id] = {
                    "track_name": test["music_data"]["spotify_data"]["track_name"],
                    "artist": test["music_data"]["spotify_data"]["artist"],
                    "artist_id": test["music_data"]["spotify_data"]["artist_id"],
                    "track_id": track_id
                }

#Build weighted vectors
parsed_songs = {}
emotion_totals = Counter()

for track_id, labels in track_labels.items():
    count = len(labels)
    counter = Counter(labels)

    #Initialize vector
    vector = [0.0] * len(GEMS_9)

    #Weighted distribution
    for i, emotion in enumerate(GEMS_9):
        if count > 0:
            vector[i] = counter[emotion] / count
            emotion_totals[emotion] += vector[i]

    parsed_songs[track_id] = {
        "music_data": track_metadata[track_id],
        "gems_vector": dict(zip(GEMS_9, vector))
    }

#Save consolidated dataset
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(parsed_songs, f, indent=2, ensure_ascii=False)

print(f"Parsed songs with weighted GEMS-9 vectors saved to {output_file}")

#Print weighted totals across dataset
print("\nWeighted totals across dataset (fractions included):")
for emotion in GEMS_9:
    print(f"- {emotion}: {emotion_totals[emotion]:.2f}")

print(f"\nUnique songs in dataset: {len(parsed_songs)}")
