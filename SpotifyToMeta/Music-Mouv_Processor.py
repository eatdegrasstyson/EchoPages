import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
music_mouv_folder = os.path.join(project_root, "DataSets", "Music-Mouv", "music-mouv")
parsed_data_folder = os.path.join(project_root, "DataSets", "Music-Mouv", "ParsedData")

#Create ParsedData folder if it doesn't exist
os.makedirs(parsed_data_folder, exist_ok=True)

participants_tests = {}

#Loop through all files in the Music-Mouv dataset
for filename in os.listdir(music_mouv_folder):
    if not filename.lower().endswith(".json"):
        continue
    
    filepath = os.path.join(music_mouv_folder, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        participant = json.load(f)

    tests_list = []

    for test_key, test in participant.items():
        if not isinstance(test, dict):
            continue

        interview = test.get("interview")
        music_data = test.get("music_data")
        if not (interview and music_data):
            continue

        #Extract only the GEMS labels
        gem_label = interview.get("standardized_esthetic_emotion")
        if not gem_label or gem_label in ["Energy", "Neutral"]:
            continue

        if gem_label == "Joyful Activation":
            gem_label = "Joy"
        
        #Extract only desired music fields
        spotify_data = music_data.get("spotify_data")
        music_style = music_data.get("music_style")
        if not spotify_data or not music_style:
            continue


        #Simplified test dictionary
        simplified_test = {
            "standardized_esthetic_emotion": gem_label,
            "music_data": {
                "spotify_data": {
                    "track_name": spotify_data.get("track_name"),
                    "track_id": spotify_data.get("track_id"),
                    "artist": spotify_data.get("artist"),
                    "artist_id": spotify_data.get("artist_id"),
                },
                "music_style": music_style
            }
        }

        tests_list.append(simplified_test)

    participants_tests[filename] = tests_list

#Save to a single JSON file
output_file = os.path.join(parsed_data_folder, "participants_tests_simplified.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(participants_tests, f, indent=2, ensure_ascii=False)

print(f"Parsed simplified data saved to {output_file}")
