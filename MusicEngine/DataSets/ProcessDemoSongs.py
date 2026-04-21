import csv
import os
import numpy as np
from pathlib import Path

from SpotifyToSpectrogram.mp3ToSpectrogram import audio_to_logmel_array
from SpotifyToSpectrogram.name_to_audio import download_mp3_from_spotify_id
from SpotifyToSpectrogram.get_metadata import get_data_from_id

INPUT_CSV  = "DataSets/CustomDataset.csv"
OUTPUT_CSV = "DataSets/CustomDatasetOut.csv"

AUDIO_DIR = Path("DataSets/audio")
SPEC_DIR  = Path("DataSets/img")

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)

def process_csv():
    rows_out = []

    with open(INPUT_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            spotify_id = row["spotify_id"]
            emotion = row["gems9_emotion"]

            print(f"\n[{i}] Processing {spotify_id} ({emotion})")

            try:
                # 1. Get metadata + download mp3
                track = get_data_from_id(spotify_id)
                mp3_path = download_mp3_from_spotify_id(track, AUDIO_DIR)

                if not mp3_path or not os.path.exists(mp3_path):
                    print("Failed to download")
                    continue
                new_mp3_path = AUDIO_DIR / f"{spotify_id}.mp3"
                os.replace(mp3_path, new_mp3_path)

                #Convert to spectrogram
                spec = audio_to_logmel_array(new_mp3_path)

                spec_path = SPEC_DIR / f"{spotify_id}.npy"
                np.save(spec_path, spec)

                n_mels, n_frames = spec.shape

                rows_out.append([
                    emotion,
                    str(new_mp3_path),
                    str(spec_path),
                    n_mels,
                    n_frames
                ])

                print("Done")

            except Exception as e:
                print(f"Error: {e}")
                continue

    # Write output CSV
    with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["emotion", "mp3_path", "spec_path", "n_mels", "n_frames"])
        writer.writerows(rows_out)

    print(f"\nFinished. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_csv()