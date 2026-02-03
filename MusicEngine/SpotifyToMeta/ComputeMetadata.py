import os
import sys
import json
import csv
import librosa
import music21
import numpy as np
import pandas as pd

def compute_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    #Tempo
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    tempo_variance = np.var(np.diff(beat_times)) if len(beat_times) > 1 else 0.0

    #RMS / Loudness / Energy
    rms = librosa.feature.rms(y=y)[0]
    loudness = np.mean(librosa.amplitude_to_db(rms))
    energy = float(np.mean(rms) / np.max(rms))  # normalized 0-1
    

    #Spectral features
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    #Chroma & Key/Mode
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_idx = int(np.argmax(np.mean(chroma, axis=1)))
    chroma_mean = np.mean(chroma, axis=1)
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    stream = music21.stream.Stream()
    for i, intensity in enumerate(chroma_mean):
        n = music21.note.Note(notes[i])
        n.volume.velocity = int(intensity * 127)
        stream.append(n)

    key_analysis = stream.analyze('Krumhansl')
    mode_name = key_analysis.mode 
    mode = 1 if mode_name == "major" else 0

    #Heuristic features
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    acousticness = np.clip(1 - spectral_flatness, 0, 1)
    danceability = np.clip((energy + tempo/200 + (1 - spectral_flatness))/3, 0, 1)
    valence = np.clip((energy + (spectral_centroid/5000) + mode)/3, 0, 1)

    features = {
        "tempo": tempo,
        "tempo_variance": tempo_variance,
        "loudness": loudness,
        "energy": energy,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "zero_crossing_rate": zero_crossing_rate,
        "key": key_idx,
        "mode": mode,
        "acousticness": acousticness,
        "danceability": danceability,
        "valence": valence,
    }

    return features

#add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from SpotifyToSpectrogram.get_metadata import get_audio_features

#paths
csv_gems = os.path.join(project_root, "DataSets", "SpotifyMetaToGems_Final.csv")
csv_mappings = os.path.join(project_root, "DataSets", "SpotifyIDMappings.csv")

#read CSVs
df_gems = pd.read_csv(csv_gems, dtype={"spotifyid": str})
df_mappings = pd.read_csv(csv_mappings, dtype={"spotifyid": str})

#check same number of rows
if len(df_gems) != len(df_mappings):
    raise ValueError(f"Row count mismatch: gems={len(df_gems)}, mappings={len(df_mappings)}")

#check ids line by line and combine
combined_rows = []
for i, (row_gems, row_map) in enumerate(zip(df_gems.itertuples(index=False), df_mappings.itertuples(index=False))):
    sid_gems = getattr(row_gems, "spotifyid")
    sid_map = getattr(row_map, "spotifyid")

    if sid_gems != sid_map:
        raise ValueError(f"Row {i} mismatch: {sid_gems} != {sid_map}")

    #merge dicts, keep all columns but duplicate spotifyid
    row_dict = {**df_gems.iloc[i].to_dict(), **{k: v for k, v in df_mappings.iloc[i].to_dict().items() if k != "spotifyid"}}
    combined_rows.append(row_dict)

#build final dataframe
combined_df = pd.DataFrame(combined_rows)

#save combined CSV
combined_csv_path = os.path.join(project_root, "DataSets", "Combined_Spotify_Meta_GEMS.csv")
combined_df.to_csv(combined_csv_path, index=False, encoding="utf-8")


all_rows = []

tmp = 0
for _, row in combined_df.iterrows():
    tmp+=1
    print(tmp)
    
    audio_path = row.get("mp3")

    #Skip if mp3 path is missing or NaN
    if not audio_path or pd.isna(audio_path):
        #print(f"Skipping {row.get('spotifyid', 'unknown ID')} — no audio path")
        continue

    audio_path = os.path.join(project_root, row["mp3"])
    if not os.path.exists(audio_path):
        print(f"Skipping missing file: {audio_path}")
        continue

    features = compute_audio_features(audio_path)
    merged_row = {**row.to_dict(), **features}
    all_rows.append(merged_row)

final_df = pd.DataFrame(all_rows)
final_csv_path = os.path.join(project_root, "DataSets", "FINALDATASET.csv")
final_df.to_csv(final_csv_path, index=False, encoding="utf-8")

