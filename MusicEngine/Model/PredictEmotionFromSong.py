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

    print("\n=== Raw Windows ===")
    for i, vec in enumerate(preds):
        top3_idx = vec.argsort()[-3:][::-1].astype(int)
        top3_emotions = [EMOTIONS[j] for j in top3_idx]
        start_time = times[i]
        end_time = start_time + WINDOW_SEC
        duration = end_time - start_time
        print(f"[{i}] {start_time:6.1f}s → {end_time:6.1f}s ({duration:5.1f}s) | {', '.join(top3_emotions)}")

    IMMEDIATE_CHANGE = 0.5
    STRONG_CHANGE    = 0.35
    MEDIUM_CHANGE    = 0.2
    WEAK_CHANGE      = 0.12

    MIN_PERSIST = 4
    DRIFT_LIMIT = 0.9
    MIN_SECONDS = 8.0

    dt = times[1] - times[0]
    min_frames = int(MIN_SECONDS / dt)
    WINDOW_SEC = FIX_FRAMES * hop_size / sr

    chunks = []
    cur_start_idx = 0

    def should_split(start_idx, lookahead=8):
        persist_count = 0
        drift_accum = 0.0
        tmp = 0
        for i in range(start_idx, min(len(preds),start_idx+lookahead)):
            tmp = tmp + 1
            cur_mean = np.mean(preds[cur_start_idx:i], axis=0)
            v = cosine_dist(preds[i], cur_mean)
            drift_accum += v

            if v > IMMEDIATE_CHANGE and tmp <= 1:
                return True, i
            elif v > STRONG_CHANGE and tmp <= 3:
                persist_count += 1
                if persist_count >= 2:
                    return True, start_idx
            elif v > MEDIUM_CHANGE and tmp <= 6:
                persist_count += 1
                if persist_count >= MIN_PERSIST:
                    return True, start_idx
            elif v > WEAK_CHANGE and tmp <= 8:
                persist_count += 1
                if drift_accum >= DRIFT_LIMIT:
                    return True, start_idx
            else:
                persist_count = max(persist_count - 1, 0)
                drift_accum *= 0.9
 
        return False, None

    i = 1
    while i < len(preds):
        split = False
        split_idx = None

        #Check if the current frame is different enough to consider a split
        v = cosine_dist(preds[i], np.mean(preds[cur_start_idx:i], axis=0))
        if v > WEAK_CHANGE:
            split, split_idx = should_split(i)

        # enforce minimum chunk size
        if split and (split_idx - cur_start_idx) < min_frames:
            split = False
            split_idx = None

        if split:
            # Compute mean for the chunk up to the split_idx
            mean_vec = np.mean(preds[cur_start_idx:split_idx], axis=0)
            top3_indices = mean_vec.argsort()[-3:][::-1]

            chunks.append({
                "start": times[cur_start_idx],
                "end": times[split_idx-1] + WINDOW_SEC,
                "emotion": mean_vec,
                "label": [EMOTIONS[j] for j in top3_indices]
            })

            # reset state for next chunk
            cur_start_idx = split_idx
            i = split_idx  # continue from the split
        else:
            i += 1

    # final chunk
    mean_vec = np.mean(preds[cur_start_idx:], axis=0)
    top3_indices = mean_vec.argsort()[-3:][::-1]
    chunks.append({
        "start": times[cur_start_idx],
        "end": times[-1] + WINDOW_SEC,
        "emotion": mean_vec,
        "label": [EMOTIONS[j] for j in top3_indices]
    })

    # print chunks
    print("\n=== Emotion Chunks ===")
    for idx, c in enumerate(chunks):
        dur = c["end"] - c["start"]
        print(f"[{idx}] {c['start']:6.1f}s → {c['end']:6.1f}s ({dur:5.1f}s) | {', '.join(c['label'])}")

    return chunks



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

    spec = audio_to_logmel_array("Temp/audio/Klaus Badelt - He's a Pirate.mp3")
    preds, times = predict_spectrogram(spec, chunking=True)
    chunkingData(preds, times)
