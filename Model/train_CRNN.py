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
    """
    Per-clip standardization (zero mean, unit variance).
    """
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
    """
    Load CRNN architecture from JSON and weights from H5.
    This avoids any issues with custom loss functions.
    """
    if not os.path.isfile(arch_path):
        raise FileNotFoundError(f"Architecture JSON not found at {arch_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    with open(arch_path) as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights(weights_path)
    return model


# =========================
# Full-song multi-window prediction
# =========================
def predict_emotion_full_song(model: keras.Model,
                              spec: np.ndarray,
                              fix_frames: int = FIX_FRAMES,
                              stride: int | None = None) -> np.ndarray:
    assert spec.ndim == 2 and spec.shape[0] == N_MELS, \
        f"Expected spectrogram of shape [N_MELS, time], got {spec.shape}"

    if stride is None:
        stride = fix_frames // 2  # 50% overlap by default

    T = spec.shape[1]

    # If song shorter than a window, just do the usual crop/pad once
    if T <= fix_frames:
        x = tf.convert_to_tensor(spec, dtype=tf.float32)
        x = _random_crop_or_pad(x, fix_frames)   # pad/crop
        x = _standardize(x)
        x = tf.expand_dims(x, -1)                # [mels, frames, 1]
        x = tf.expand_dims(x, 0)                 # [1, mels, frames, 1]
        return model.predict(x, verbose=0)[0]    # [9]

    # Build sliding window start indices
    starts = list(range(0, T - fix_frames + 1, stride))
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
    return preds.mean(axis=0)                    # [9]


# =========================
# End-to-end pipeline
# =========================
def predict_emotion(spotify_id: str):
    print(f"Fetching metadata for Spotify ID: {spotify_id}")
    track = get_data_from_id(spotify_id)  # (artist, title, duration_ms, ...)
    mp3_path = download_mp3_from_spotify_id(track, TMP_AUDIO_PATH)

    if not mp3_path or not os.path.exists(mp3_path):
        print("Failed to download or locate audio.")
        return

    print(f"Downloaded audio: {mp3_path}")

    # Convert to spectrogram array
    print("Converting audio to log-mel spectrogram...")
    spec = audio_to_logmel_array(mp3_path)  # [N_MELS, time]
    spec_npy_path = TMP_SPEC_PATH / f"{Path(mp3_path).stem}.npy"
    np.save(spec_npy_path, spec)
    print(f"Saved spectrogram array to: {spec_npy_path}")

    # Load model
    print("Loading model from JSON + weights...")
    model = load_model_from_json_and_weights()

    # Predict over full song via multi-window averaging
    print("Predicting emotion over full song (multi-window averaging)...")
    probs = predict_emotion_full_song(model, spec, fix_frames=FIX_FRAMES, stride=FIX_FRAMES // 2)  # 50% overlap

    # Show results sorted by confidence
    results = sorted(zip(EMOTIONS, probs), key=lambda kv: kv[1], reverse=True)
    print("\n=== Emotion Prediction ===")
    for emo, p in results:
        print(f"{emo:15s}: {p:.3f}")

    top_emo, top_p = results[0]
    print(f"\nDominant emotion: {top_emo} ({top_p:.2%})")


# =========================
# CLI entry
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m YourModule.predict <spotify_id>")
        sys.exit(1)

    spotify_id = sys.argv[1]
    predict_emotion(spotify_id)
