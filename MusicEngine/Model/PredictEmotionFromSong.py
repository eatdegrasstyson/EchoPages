import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from SpotifyToSpectrogram.mp3ToSpectrogram import audio_to_logmel_array
from SpotifyToSpectrogram.name_to_audio import download_mp3_from_spotify_id
from SpotifyToSpectrogram.get_metadata import get_data_from_id

# =========================
# Config
# =========================
ARCH_PATH    = "Model/crnn_arch.json"
WEIGHTS_PATH = "Model/crnn_weights.h5"

TMP_AUDIO_PATH = Path("Temp/audio")
TMP_SPEC_PATH  = Path("Temp/spec")

N_MELS     = 128
FIX_FRAMES = 512
STRIDE_FRAMES = FIX_FRAMES // 2   # match training overlap (50%)
EMOTIONS   = [
    "Wonder","Transcendence","Tenderness","Nostalgia","Peacefulness",
    "Power","Joy","Tension","Sadness"
]

TMP_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
TMP_SPEC_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# Preprocessing (same as training)
# =========================
def _random_crop_or_pad(x: tf.Tensor, target_frames: int = FIX_FRAMES) -> tf.Tensor:
    """
    x: [mels, time]
    If time < target_frames: right-pad with zeros.
    If time >= target_frames: random crop of length target_frames.
    """
    t = tf.shape(x)[1]

    def pad():
        pad_amt = target_frames - t
        return tf.pad(x, [[0, 0], [0, pad_amt]])

    def crop():
        start = tf.random.uniform((), 0, t - target_frames + 1, dtype=tf.int32)
        return x[:, start:start + target_frames]

    return tf.cond(t < target_frames, pad, crop)


def _standardize(x: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(x)
    std  = tf.math.reduce_std(x) + 1e-6
    return (x - mean) / std


# =========================
# Model loading
# =========================
def load_model_from_json_and_weights(
    arch_path: str = ARCH_PATH,
    weights_path: str = WEIGHTS_PATH,
) -> keras.Model:
    if not os.path.isfile(arch_path):
        raise FileNotFoundError(f"Architecture JSON not found at {arch_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    with open(arch_path) as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights(weights_path)
    return model

def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred) * 0.0

def load_saved_model(path: str) -> keras.Model:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    
    # Load the model with a placeholder for the custom weighted KL loss
    return keras.models.load_model(
        path,
        custom_objects={"weighted_kl": dummy_loss}  # match name used in training
    )


# =========================
# Full-song multi-window prediction
# =========================
def predict_emotion_full_song(model: keras.Model,
                              spec: np.ndarray,
                              fix_frames: int = FIX_FRAMES,
                              stride_frames: int = STRIDE_FRAMES,
                              chunking: bool = False) -> tuple[np.ndarray,np.ndarray]:
    """
    spec: np.ndarray [n_mels, time]
    Returns: np.ndarray [9] averaged emotion distribution over the whole song.

    Matches training logic: slide fixed windows across entire song and treat
    each window as a training example. At inference we average window predictions.
    """
    if spec.ndim != 2 or spec.shape[0] != N_MELS:
        raise ValueError(f"Expected spectrogram shape [N_MELS, time], got {spec.shape}")

    T = spec.shape[1]

    # If the song is shorter than one window, just pad/crop once.
    if T <= fix_frames:
        x = tf.convert_to_tensor(spec, dtype=tf.float32)
        x = _random_crop_or_pad(x, fix_frames)   # pad/crop to FIX_FRAMES
        x = _standardize(x)
        x = tf.expand_dims(x, -1)                # [mels, frames, 1]
        x = tf.expand_dims(x, 0)                 # [1, mels, frames, 1]
        return model.predict(x, verbose=0)[0]

    # Build sliding window start indices
    starts = list(range(0, T - fix_frames + 1, stride_frames))
    if not starts:
        starts = [0]

    preds = []
    for s in starts:
        window = spec[:, s:s + fix_frames]       # [mels, FIX_FRAMES]
        x = tf.convert_to_tensor(window, dtype=tf.float32)
        x = _standardize(x)
        x = tf.expand_dims(x, -1)                # [mels, frames, 1]
        x = tf.expand_dims(x, 0)                 # [1, mels, frames, 1]
        p = model.predict(x, verbose=0)[0]       # [9]
        preds.append(p)

    preds = np.stack(preds, axis=0)              # [num_windows, 9]
    
    # assumes: hop_size is same as stride_frames in frames
    # and spectrogram was created at sr=32000 Hz
    sr = 32000
    hop_size = 320  # must match audio_to_logmel_array hop
    chunk_times = np.array(starts) * hop_size / sr


    if not chunking:
        return preds.mean(axis=0),chunk_times                 # [9], averaged distribution
    else:
        return preds,chunk_times


# =========================
# End-to-end prediction
# =========================
def predict_emotion(spotify_id: str):
    # 1. Download audio via YouTube based on Spotify metadata
    track = get_data_from_id(spotify_id)
    mp3_path = download_mp3_from_spotify_id(track, TMP_AUDIO_PATH)

    if not mp3_path or not os.path.exists(mp3_path):
        print("Failed to download or locate audio.")
        return

    # 2. Convert audio to log-mel spectrogram
    spec = audio_to_logmel_array(mp3_path)  # [N_MELS, time]
    npy_path = TMP_SPEC_PATH / f"{Path(mp3_path).stem}.npy"
    np.save(npy_path, spec)

    # 3. Load model
    print("Loading model...")
    model = load_saved_model("Model/epoch_09.h5")

    # 4. Predict over full song via sliding windows + averaging
    print("Predicting emotion over full song (sliding windows)...")
    preds,_ = predict_emotion_full_song(model, spec)

    # 5. Show results
    results = sorted(zip(EMOTIONS, preds), key=lambda kv: kv[1], reverse=True)
    print("\n=== Emotion Prediction ===")
    for emo, p in results:
        print(f"{emo:15s}: {p:.3f}")

    top_emo, top_p = results[0]
    print(f"\nDominant emotion: {top_emo} ({top_p:.2%})")


def predict_spectrogram(spec, chunking=False):
    # 3. Load model
    print("Loading model...")
    model = load_saved_model("Model/epoch_09.h5")
    preds,times = predict_emotion_full_song(model, spec, chunking=chunking)
    return preds,times

def cosine_dist(a, b, eps=1e-8):
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

def chunkingData(preds, times):
    hop_size = 320
    sr = 32000
    WINDOW_SEC = FIX_FRAMES * hop_size / sr

    #Dynamic params:

    SMOOTH_WINDOW = 5
    COMPARE_WINDOW = 6

    #Smoothed predictions
    smoothed = []
    for i in range(len(preds)):
        start = max(0, i - SMOOTH_WINDOW)
        end = min(len(preds), i + SMOOTH_WINDOW + 1)
        smoothed.append(np.mean(preds[start:end], axis=0))
    preds = np.array(smoothed)

    structural_dists = []

    for i in range(COMPARE_WINDOW, len(preds) - COMPARE_WINDOW):
        past_mean = np.mean(preds[i-COMPARE_WINDOW:i], axis=0)
        future_mean = np.mean(preds[i:i+COMPARE_WINDOW], axis=0)

        v = cosine_dist(past_mean, future_mean)
        structural_dists.append(v)

    structural_dists = np.array(structural_dists)

    #Reduce num for more chunks, increase for less chunks
    chunkMultiplier = 85
    SPLIT_THRESHOLD = np.percentile(structural_dists, chunkMultiplier)

    MIN_SECONDS = 18.0
    MERGE_THRESHOLD = 0.12

    dt = times[1] - times[0]
    min_frames = int(MIN_SECONDS / dt)

    # Detect structural change via past vs future window comparison
    candidate_splits = []
    for i in range(COMPARE_WINDOW, len(preds) - COMPARE_WINDOW):
        past_mean = np.mean(preds[i-COMPARE_WINDOW:i], axis=0)
        future_mean = np.mean(preds[i:i+COMPARE_WINDOW], axis=0)

        v = cosine_dist(past_mean, future_mean)

        if v > SPLIT_THRESHOLD:
            candidate_splits.append(i)

    # Enforce min distance between splits
    filtered_splits = []
    last_split = -9999
    for s in candidate_splits:
        if s - last_split >= min_frames:
            filtered_splits.append(s)
            last_split = s

    # Build chunks from filtered splits
    chunks = []
    split_points = [0] + filtered_splits + [len(preds)]

    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i+1]

        mean_vec = np.mean(preds[start_idx:end_idx], axis=0)
        top3_indices = mean_vec.argsort()[-9:][::-1]

        chunks.append({
            "start": times[start_idx],
            "end": times[end_idx-1] + WINDOW_SEC,
            "emotion": mean_vec,
            "label": [(EMOTIONS[j], mean_vec[j]) for j in top3_indices]
        })

    # Merge emotionally similar adjacent chunks
    merged = [chunks[0]]
    for c in chunks[1:]:
        prev = merged[-1]
        if cosine_dist(prev["emotion"], c["emotion"]) < MERGE_THRESHOLD:
            prev["end"] = c["end"]
            prev["emotion"] = (prev["emotion"] + c["emotion"]) / 2
        else:
            merged.append(c)

    for idx, c in enumerate(merged):
        dur = c["end"] - c["start"]
        top3_str = ", ".join([f"{e} ({v:.2f})" for e, v in c['label']])
        print(f"[{idx}] {c['start']:6.1f}s → {c['end']:6.1f}s ({dur:5.1f}s) | {top3_str}")

    return merged




# =========================
# CLI entry
# =========================
if __name__ == "__main__":
    '''
    if len(sys.argv) < 2:
        print("Usage: python -m YourModule.predict <spotify_id>")
        sys.exit(1)

    spotify_id = sys.argv[1]
    predict_emotion(spotify_id)
    '''
    song_name = "Baby Keem - family ties (with Kendrick Lamar)"
    path = f"Temp/audio/{song_name}.mp3"
    spotify_id = "7tFiyTwD0nx5a1eklYtX2J"
    if not os.path.exists(path):
        predict_emotion(spotify_id)

    spec = audio_to_logmel_array(path)
    preds, times = predict_spectrogram(spec, chunking=True)

    print("Chat GPT algorithm idea: ")
    chunkingData(preds, times)

