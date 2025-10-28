import os
import sys
import json
import csv
import librosa
import music21
import numpy as np

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


csv_path = os.path.join(project_root, "DataSets", "SpotifyMetaToGems_Final.csv")
audio_path = os.path.join(project_root, "DataSets", "audio", "Junior Senior - Move Your Feet.mp3")

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

first_row = rows[0]
spotify_id = first_row["spotifyid"]
gems_vector = [
    float(first_row["Wonder"]),
    float(first_row["Transcendence"]),
    float(first_row["Tenderness"]),
    float(first_row["Nostalgia"]),
    float(first_row["Peacefulness"]),
    float(first_row["Power"]),
    float(first_row["Joy"]),
    float(first_row["Tension"]),
    float(first_row["Sadness"]),
]

print("Spotify ID:", spotify_id)
print("GEMS-9 Vector:", gems_vector)
features = compute_audio_features(audio_path)
for k, v in features.items():
    print(k, v)


