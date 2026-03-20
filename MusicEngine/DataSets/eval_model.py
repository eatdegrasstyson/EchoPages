"""
CLI evaluator for any trained checkpoint against the Emotify test set.
Mirrors testmetrics.py but accepts --model to specify which .h5 to load.

Usage (from MusicEngine/ directory):
    python -m DataSets.eval_model --model Model/cnn_main_final.h5
    python -m DataSets.eval_model --model Model/crnn_deam_final.h5
    python -m DataSets.eval_model --model Model/cnn_epoch_10.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.linalg import norm
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent.parent))
from SpotifyToSpectrogram.mp3ToSpectrogram import audio_to_logmel_array

FIX_FRAMES    = 512
STRIDE_FRAMES = FIX_FRAMES // 2
N_MELS        = 128

EMOTIONS = [
    "Wonder", "Transcendence", "Tenderness", "Nostalgia", "Peacefulness",
    "Power", "Joy", "Tension", "Sadness"
]

TMP_SPEC_PATH = Path("Emotify/Spectrograms")
TMP_SPEC_PATH.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = Path(__file__).parent


# ============================================================
# Preprocessing helpers
# ============================================================
def _random_crop_or_pad(x: tf.Tensor, target_frames: int = FIX_FRAMES) -> tf.Tensor:
    t = tf.shape(x)[1]
    def pad():  return tf.pad(x, [[0, 0], [0, target_frames - t]])
    def crop():
        start = tf.random.uniform((), 0, t - target_frames + 1, dtype=tf.int32)
        return x[:, start:start + target_frames]
    return tf.cond(t < target_frames, pad, crop)


def _standardize(x: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(x)
    std  = tf.math.reduce_std(x) + 1e-6
    return (x - mean) / std


# ============================================================
# Inference
# ============================================================
def predict_emotion_full_song(
    model: keras.Model,
    spec: np.ndarray,
    fix_frames: int = FIX_FRAMES,
    stride_frames: int = STRIDE_FRAMES,
) -> np.ndarray:
    if spec.ndim != 2 or spec.shape[0] != N_MELS:
        raise ValueError(f"Expected spectrogram shape [N_MELS, time], got {spec.shape}")

    T = spec.shape[1]
    if T <= fix_frames:
        x = tf.convert_to_tensor(spec, dtype=tf.float32)
        x = _random_crop_or_pad(x, fix_frames)
        x = _standardize(x)
        x = tf.expand_dims(tf.expand_dims(x, -1), 0)
        return model.predict(x, verbose=0)[0]

    starts = list(range(0, T - fix_frames + 1, stride_frames)) or [0]
    preds  = []
    for s in starts:
        window = spec[:, s:s + fix_frames]
        x = tf.convert_to_tensor(window, dtype=tf.float32)
        x = _standardize(x)
        x = tf.expand_dims(tf.expand_dims(x, -1), 0)
        preds.append(model.predict(x, verbose=0)[0])
    return np.stack(preds, axis=0).mean(axis=0)


# ============================================================
# Metrics
# ============================================================
def cosine_similarity(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))

def top1_accuracy(true_vec, pred_vec):
    return int(np.argmax(true_vec) == np.argmax(pred_vec))

def top3_accuracy(true_vec, pred_vec):
    true_top3 = set(np.argsort(true_vec)[-3:])
    pred_top3 = set(np.argsort(pred_vec)[-3:])
    return int(len(true_top3 & pred_top3) > 0)


def get_song_path(song_id: int, genre: str) -> str:
    offset = {"classical": 0, "rock": 100, "electronic": 200, "pop": 300}
    g = genre.lower()
    if g not in offset:
        raise ValueError(f"Unsupported genre: {genre}")
    new_id = song_id - offset[g]
    return f"DataSets/Emotify/{genre}/{new_id}.mp3"


# ============================================================
# Main evaluation loop
# ============================================================
def evaluate(model_path: str) -> None:
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    csv_path = SCRIPT_DIR / "testset.csv"
    df = pd.read_csv(csv_path)

    genre_stats: dict = {}
    overall_cos, overall_top1, overall_top3 = [], [], []
    total = len(df)

    for idx, row in df.iterrows():
        true_vec  = row[EMOTIONS].values.astype(float)
        song_id   = int(row["track_id"])
        genre     = row["genre"]
        song_path = get_song_path(song_id, genre)

        spec         = audio_to_logmel_array(song_path)
        npy_filename = f"{genre}_{Path(song_path).stem}.npy"
        npy_path     = TMP_SPEC_PATH / npy_filename
        np.save(npy_path, spec)

        pred_vec = predict_emotion_full_song(model, spec)

        cos = cosine_similarity(true_vec, pred_vec)
        t1  = top1_accuracy(true_vec, pred_vec)
        t3  = top3_accuracy(true_vec, pred_vec)

        overall_cos.append(cos)
        overall_top1.append(t1)
        overall_top3.append(t3)

        if genre not in genre_stats:
            genre_stats[genre] = {"cos": [], "top1": [], "top3": []}
        genre_stats[genre]["cos"].append(cos)
        genre_stats[genre]["top1"].append(t1)
        genre_stats[genre]["top3"].append(t3)

        remaining = total - (idx + 1)
        print(
            f"Processed {idx + 1}/{total} ({remaining} remaining) | "
            f"Cos: {cos:.4f}, Top1: {t1}, Top3: {t3}"
        )

    print("\n===== OVERALL METRICS =====")
    print(f"Cosine similarity: {np.mean(overall_cos):.4f}")
    print(f"Top-1 Accuracy   : {np.mean(overall_top1):.4f}")
    print(f"Top-3 Accuracy   : {np.mean(overall_top3):.4f}")

    print("\n===== METRICS BY GENRE =====")
    for genre, stats in genre_stats.items():
        print(f"\nGenre: {genre}")
        print(f"  Cosine similarity: {np.mean(stats['cos']):.4f}")
        print(f"  Top-1 Accuracy   : {np.mean(stats['top1']):.4f}")
        print(f"  Top-3 Accuracy   : {np.mean(stats['top3']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the Emotify test set.")
    parser.add_argument("--model", required=True, help="Path to .h5 model file (e.g. Model/cnn_main_final.h5)")
    args = parser.parse_args()
    evaluate(args.model)
