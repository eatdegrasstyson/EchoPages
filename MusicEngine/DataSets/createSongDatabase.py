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


csv_path = "DataSets/SpotifyIDMappings.csv"
df = pd.read_csv(csv_path)

#Assuming column names: spotifyid, mp3, spec_npy, n_mels, n_frames
spec_paths = df['spec_npy'].tolist()
spotify_ids = df['spotifyid'].tolist()

#Grabs the name of the song for good visuals (author too)
def extract_song_text(mp3_path):
    return Path(mp3_path).stem


def process_dataset(test_first_only=False, numSongs=0):
    paths_to_process = spec_paths[:numSongs] if test_first_only else spec_paths
    results = predict_multiple_songs(paths_to_process, chunking=True)

    #Averaged CSV
    averaged_csv_path = "DataSets/songDataBaseAveraged.csv"
    with open(averaged_csv_path, 'w', newline='') as f_avg:
        writer = csv.writer(f_avg)
        header = ['spotifyID', 'song_name'] + EMOTIONS
        writer.writerow(header)

        for (spec_path, avg_vec, _) in results:
            song_name = extract_song_text(spec_path)
            spotify_id = df.loc[df['spec_npy'] == spec_path, 'spotifyid'].values[0]
            row = [spotify_id, song_name] + avg_vec.tolist()
            writer.writerow(row)

    #Chunked CSV
    chunked_csv_path = "DataSets/songDataBaseChunked.csv"
    with open(chunked_csv_path, 'w', newline='') as f_chunk:
        writer = csv.writer(f_chunk)
        header = ['spotifyID', 'song_name', 'start', 'end'] + EMOTIONS
        writer.writerow(header)

        for (spec_path, _, chunks) in results:
            song_name = extract_song_text(spec_path)
            spotify_id = df.loc[df['spec_npy'] == spec_path, 'spotifyid'].values[0]
            for c in chunks:
                row = [spotify_id, song_name, c['start'], c['end']] + c['emotion'].tolist()
                writer.writerow(row)


if __name__ == "__main__":
    # test only first song
    process_dataset(test_first_only=True, numSongs=3)