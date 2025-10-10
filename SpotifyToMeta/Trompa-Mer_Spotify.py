import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

input_file = os.path.join(project_root, "DataSets", "Trompa-Mer", "ParsedData", "filtered_trompa_mer.csv")
output_file = os.path.join(project_root, "DataSets", "Trompa-Mer", "ParsedData", "trompa_mer_with_spotify_ids.csv")

# Spotify credentials from your environment variables
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

df = pd.read_csv(input_file)

spotify_ids = []

for _, row in df.iterrows():
    track_name = row["track_name"]
    artist_name = row.get("artist_name", "")
    composer_name = row.get("composer_name", "")

    query_parts = [f'track:"{track_name}"']
    if artist_name.lower() != "not found":
        query_parts.append(f'artist:"{artist_name}"')
    elif composer_name.lower() != "not found":
        query_parts.append(f'artist:"{composer_name}"')

    query = " ".join(query_parts)

    try:
        results = sp.search(q=query, type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])
        if items:
            spotify_ids.append(items[0]["id"])
        else:
            spotify_ids.append(None)
    except Exception as e:
        print(f"Error searching for {track_name}: {e}")
        spotify_ids.append(None)

df["spotify_track_id"] = spotify_ids
df.to_csv(output_file, index=False)
print(f" Spotify IDs added and saved to {output_file}")
