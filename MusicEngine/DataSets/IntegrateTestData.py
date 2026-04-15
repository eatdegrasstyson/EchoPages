import pandas as pd
import numpy as np
from pathlib import Path
import os

from SpotifyToSpectrogram.mp3ToSpectrogram import audio_to_logmel_array

#Emotify path to data
TESTSET_CSV = "DataSets/testset.csv"

GEMS_CSV = "DataSets/SpotifyMetaToGems_Final.csv"
MAPPING_CSV = "DataSets/SpotifyIDMappings.csv"

IMG_OUTPUT_PATH = Path("DataSets/img")
IMG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

EMOTIONS = [
    "Wonder","Transcendence","Tenderness","Nostalgia","Peacefulness",
    "Power","Joy","Tension","Sadness"
]

def get_song_path(song_id, genre):
    if genre.lower() == "classical":
        new_id = song_id
    elif genre.lower() == "rock":
        new_id = song_id - 100
    elif genre.lower() == "electronic":
        new_id = song_id - 200
    elif genre.lower() == "pop":
        new_id = song_id - 300
    else:
        raise ValueError("Unsupported genre: " + genre)

    return f"DataSets/Emotify/{genre}/{new_id}.mp3"

gems_df_existing = pd.read_csv(GEMS_CSV)
mapping_df_existing = pd.read_csv(MAPPING_CSV)

existing_ids = set(gems_df_existing["spotifyid"])

#testset
df = pd.read_csv(TESTSET_CSV)

gems_rows = []
mapping_rows = []

for idx, row in df.iterrows():
    track_id = int(row["track_id"])
    genre = row["genre"]

    # ✅ Unique ID (no collision with Spotify)
    uid = track_id

    if uid in existing_ids:
        print(f"Skipping duplicate: {uid}")
        continue

    song_path = get_song_path(track_id, genre)

    if not os.path.isfile(song_path):
        print(f"Missing file: {song_path}")
        continue

    #Generate spectrogram
    spec = audio_to_logmel_array(song_path)

    #Save npy
    npy_path = IMG_OUTPUT_PATH / f"{uid}.npy"
    np.save(npy_path, spec)


    gems_row = {
        "spotifyid": uid,
        "DataSet": f"Emotify:{genre}",
    }
    for emo in EMOTIONS:
        gems_row[emo] = float(row[emo])

    
    # Mapping row

    mapping_row = {
        "spotifyid": uid,
        "mp3": song_path,
        "spec_npy": str(npy_path),
        "n_mels": int(spec.shape[0]),
        "n_frames": int(spec.shape[1]),
    }

    gems_rows.append(gems_row)
    mapping_rows.append(mapping_row)

    print(f"Added: {uid}")

#def audio_to_logmel_array(path, sr=32000, n_fft=1024, hop=320, n_mels=128):
MAX_DURATION_SECONDS = 20 * 60
HOP_LENGTH = 320
SR = 32000

if gems_rows:
    gems_df_new = pd.DataFrame(gems_rows)
    mapping_df_new = pd.DataFrame(mapping_rows)

    gems_combined = pd.concat([gems_df_existing, gems_df_new], ignore_index=True)
    mapping_combined = pd.concat([mapping_df_existing, mapping_df_new], ignore_index=True)
else:
    gems_combined = gems_df_existing.copy()
    mapping_combined = mapping_df_existing.copy()

# Compute duration from n_frames
def is_valid_length(n_frames):
    try:
        duration = (int(n_frames) * HOP_LENGTH) / SR
        return duration <= MAX_DURATION_SECONDS
    except:
        return False
    
# Filter mapping
mapping_clean = mapping_combined[
    mapping_combined["n_frames"].apply(is_valid_length)
].copy()

# Keep only valid IDs
valid_ids = set(mapping_clean["spotifyid"])

gems_clean = gems_combined[
    gems_combined["spotifyid"].isin(valid_ids)
].copy()

# Save cleaned versions
mapping_clean["n_mels"] = mapping_clean["n_mels"].astype(int)
mapping_clean["n_frames"] = mapping_clean["n_frames"].astype(int)
gems_clean.to_csv(GEMS_CSV, index=False)
mapping_clean.to_csv(MAPPING_CSV, index=False)

print(f"\nFinal dataset size: {len(valid_ids)} entries (after removing long songs)")