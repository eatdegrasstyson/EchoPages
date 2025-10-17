import os
import sys
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
spectrogram_dir = os.path.join(project_root, "SpotifyToSpectrogram")
sys.path.append(spectrogram_dir)

import SpotifyCreds  # type: ignore

creds = SpotifyCreds.SpotifyCreds()
auth_manager = SpotifyClientCredentials(client_id=creds.CLIENT, client_secret=creds.CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

input_file = os.path.join(project_root, "DataSets", "Trompa-Mer", "ParsedData", "filtered_trompa_mer.csv")
output_file = os.path.join(project_root, "DataSets", "Trompa-Mer", "ParsedData", "trompa_mer_with_spotify_ids.csv")

df = pd.read_csv(input_file)
spotify_ids = []

i = 1
for _, row in df.iterrows():
    print(i)
    track_name = row["track_name"]
    artist_name = row.get("artist_name", "")
    composer_name = row.get("composer_name", "")

    query_base = f'track:"{track_name}"'
    primary_query = None
    secondary_query = None

    if artist_name.lower() != "not found":
        primary_query = f'{query_base} artist:"{artist_name}"'
    elif composer_name.lower() != "not found":
        primary_query = f'{query_base} artist:"{composer_name}"'

    if artist_name.lower() != "not found" and composer_name.lower() != "not found":
        secondary_query = f'{query_base} artist:"{artist_name}" artist:"{composer_name}"'

    found_id = None
    try:
        if primary_query:
            results = sp.search(q=primary_query, type="track", limit=1)
            items = results.get("tracks", {}).get("items", [])
            if items:
                found_id = items[0]["id"]
        if not found_id and secondary_query:
            results = sp.search(q=secondary_query, type="track", limit=1)
            items = results.get("tracks", {}).get("items", [])
            if items:
                found_id = items[0]["id"]
    except Exception as e:
        print(f"Error searching for {track_name}: {e}")

    spotify_ids.append(found_id if found_id else "NA")
    i += 1

df["track_id"] = spotify_ids
df.to_csv(output_file, index=False)

print(f"Spotify IDs added and saved to {output_file}")
