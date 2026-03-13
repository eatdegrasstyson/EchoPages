import csv
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Model.PredictEmotionFromSong import (
    predict_spectrogram,
    chunkingData_dp,
    EMOTIONS
)

INPUT_CSV = ROOT / "DataSets" / "SpotifyIDMappings.csv"
AVG_OUT = ROOT / "DataSets" / "songDataBaseAveraged.csv"
CHUNK_OUT = ROOT / "DataSets" / "songDataBaseChunked.csv"

#Grabs the name of the song for good visuals (author too)
def extract_song_text(mp3_path):
    return Path(mp3_path).stem

#Gets the prediction given the spectrogram path if it exists.
def process_song(spec_path):
    spec = np.load(ROOT / spec_path)
    preds, times = predict_spectrogram(spec, chunking=True)

    avg_vec = preds.mean(axis=0)
    chunks = chunkingData_dp(preds, times)

    return avg_vec, chunks


def process_dataset(test_first_only=False):
    averaged_rows = []
    chunk_rows = []

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            spotify_id = row["spotifyid"]
            mp3_path = row["mp3"]
            spec_path = row["spec_npy"]

            song_text = extract_song_text(mp3_path)

            try:
                avg_vec, chunks = process_song(spec_path)
            except Exception as e:
                print(f"Skipping {spotify_id}: {e}")
                continue

            avg_row = {
                "spotifyID": spotify_id,
                "songtext": song_text
            }
            for i, emo in enumerate(EMOTIONS):
                avg_row[emo] = float(avg_vec[i])
            averaged_rows.append(avg_row)

            for chunk in chunks:
                row_data = {
                    "spotifyID": spotify_id,
                    "songtext": song_text,
                    "start": float(chunk["start"]),
                    "end": float(chunk["end"])
                }
                vec = chunk["emotion"]
                for i, emo in enumerate(EMOTIONS):
                    row_data[emo] = float(vec[i])
                chunk_rows.append(row_data)

            if test_first_only:
                break

    avg_fields = ["spotifyID", "songtext"] + EMOTIONS
    with open(AVG_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=avg_fields)
        writer.writeheader()
        writer.writerows(averaged_rows)

    chunk_fields = ["spotifyID", "songtext", "start", "end"] + EMOTIONS
    with open(CHUNK_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=chunk_fields)
        writer.writeheader()
        writer.writerows(chunk_rows)


if __name__ == "__main__":
    # test only first song
    process_dataset(test_first_only=True)
