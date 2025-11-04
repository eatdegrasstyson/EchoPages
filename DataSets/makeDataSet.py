import pandas as pd
from pathlib import Path
import numpy as np
import csv
from SpotifyToSpectrogram.get_metadata import get_data_from_id
from SpotifyToSpectrogram.name_to_audio import download_mp3_from_spotify_id
from SpotifyToSpectrogram.mp3ToSpectrogram import audio_to_logmel_array

IMG_OUTPUT_PATH = Path("DataSets/img")
AUDIO_OUTPUT_PATH = Path("DataSets/audio")
LOG_PATH = Path("DataSets/SpotifyIDMappings.csv")

IMG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)




df = pd.read_csv("DataSets/SpotifyMetaToGems_Final.csv")

rows = []

i = 0
for spotify_id in df["spotifyid"]:
    if i > 15:
        break
    i+=1
    track = get_data_from_id(spotify_id)
    full_audio_path = download_mp3_from_spotify_id(track, AUDIO_OUTPUT_PATH)

    if full_audio_path:
        base = Path(full_audio_path).stem
        npy_path = IMG_OUTPUT_PATH / f"{base}.npy"

        S_db = audio_to_logmel_array(full_audio_path)
        np.save(npy_path, S_db)
        row = {
            "spotifyid": spotify_id,
            "mp3": str(full_audio_path),
            "spec_npy": str(npy_path),
            "n_mels": S_db.shape[0],
            "n_frames": S_db.shape[1],
        }
    else:
        row = {
            "spotifyid": spotify_id,
            "mp3": None,
            "spec_npy": None,
            "n_mels": None,
            "n_frames": None,
        }

    rows.append(row)

    print(row)
    
spectrograms = pd.DataFrame(rows)
spectrograms.to_csv(LOG_PATH)