import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Model.PredictEmotionFromSong import (
    chunkingData_dp,
    predict_multiple_songs,
    EMOTIONS
)

#csv_path = "DataSets/SpotifyIDMappings.csv"
csv_path = "DataSets/CustomDatasetOut.csv"
df = pd.read_csv(csv_path)

# Assuming column names: spotifyid, mp3, spec_npy, n_mels, n_frames
spec_paths = df['spec_npy'].tolist()
spotify_ids = df['spotifyid'].tolist()

# Grabs the name of the song for good visuals (author too)
def extract_song_text(mp3_path):
    return Path(mp3_path).stem


def process_dataset(test_first_only=False, numSongs=0, start_index=0):
    paths = spec_paths[start_index:]
    if test_first_only:
        paths = paths[:numSongs]

    averaged_csv_path = "DataSets/songDataBaseAveragedDemo.csv"
    chunked_csv_path = "DataSets/songDataBaseChunkedDemo.csv"

    # Write headers only if file doesn't exist
    if not Path(averaged_csv_path).exists():
        with open(averaged_csv_path, 'w', newline='') as f_avg:
            writer = csv.writer(f_avg)
            writer.writerow(['spotifyID', 'song_name'] + EMOTIONS)

    if not Path(chunked_csv_path).exists():
        with open(chunked_csv_path, 'w', newline='') as f_chunk:
            writer = csv.writer(f_chunk)
            writer.writerow(['spotifyID', 'song_name', 'start', 'end'] + EMOTIONS)

    for i, spec_path in enumerate(paths, start=start_index):
        try:
            row_df = df.iloc[i]

            # ---- Skip large spectrograms ----
            n_frames = row_df['n_frames']
            if n_frames > 200000:
                print(f"[SKIP] Row {i} | Too many frames: {n_frames}")
                continue

            print(f"[PROCESSING] Row {i} | {spec_path}")

            # Run prediction on single song
            results = predict_multiple_songs([spec_path], chunking=True)

            for (spec_path, avg_vec, chunks) in results:
                song_name = extract_song_text(spec_path)
                spotify_id = row_df['spotifyid']

                # ---- Append averaged ----
                with open(averaged_csv_path, 'a', newline='') as f_avg:
                    writer = csv.writer(f_avg)
                    row = [spotify_id, song_name] + avg_vec.tolist()
                    writer.writerow(row)

                # ---- Append chunked ----
                with open(chunked_csv_path, 'a', newline='') as f_chunk:
                    writer = csv.writer(f_chunk)
                    for c in chunks:
                        row = [spotify_id, song_name, c['start'], c['end']] + c['emotion'].tolist()
                        writer.writerow(row)

            print(f"[DONE] Row {i}")

        except Exception as e:
            print(f"[ERROR] Row {i} | {spec_path} | {e}")
            continue


if __name__ == "__main__":
    # Example: start from row 0
    process_dataset(start_index=0)