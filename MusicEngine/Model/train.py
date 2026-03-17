"""
Unified training script. Switch MODEL_TYPE and DATASET_TYPE at the top.
Does NOT modify train_CRNN.py or PredictEmotionFromSong.py.

Usage (from MusicEngine/ directory):
    python -m Model.train
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path

# Add MusicEngine root to path so relative imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIG — edit these two lines to switch modes
# ============================================================
MODEL_TYPE   = "cnn"    # "crnn" | "cnn"
DATASET_TYPE = "main"   # "main" | "deam"

# ============================================================
# Shared hyperparameters
# ============================================================
N_MELS        = 128
FIX_FRAMES    = 512
STRIDE_FRAMES = FIX_FRAMES // 2
BATCH_SIZE    = 32
EPOCHS        = 10

EMOTIONS = [
    "Wonder", "Transcendence", "Tenderness", "Nostalgia", "Peacefulness",
    "Power", "Joy", "Tension", "Sadness"
]

# ============================================================
# "main" dataset paths (Spotify-derived, relative to MusicEngine/)
# ============================================================
MAIN_MAPPING_CSV = "DataSets/SpotifyIDMappings.csv"
MAIN_LABELS_CSV  = "DataSets/SpotifyMetaToGems_Final.csv"

# ============================================================
# DEAM dataset paths — download audio and annotations first.
# Audio:       https://zenodo.org/records/1188976
# Annotations: static_annotations_averaged_songs_1_2000.csv (same archive)
# ============================================================
DEAM_ANNOTATIONS_CSV = "DataSets/DEAM/static_annotations_averaged_songs_1_2000.csv"
DEAM_AUDIO_DIR       = "DataSets/DEAM/audio"
DEAM_SPEC_DIR        = "DataSets/DEAM/spectrograms"


# ============================================================
# Preprocessing helpers (same logic as train_CRNN.py)
# ============================================================
def _random_crop_or_pad(x: tf.Tensor, target_frames: int = FIX_FRAMES) -> tf.Tensor:
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


def load_npy_tensor(path):
    def _np_load(p):
        arr = np.load(p.decode("utf-8")).astype(np.float32, copy=False)
        if arr.ndim == 3:
            arr = arr.squeeze()
        return arr
    x = tf.numpy_function(_np_load, [path], tf.float32)
    x.set_shape([N_MELS, None])
    return x


def _map_fn(path, y, start):
    x = load_npy_tensor(path)
    t = tf.shape(x)[1]

    def use_full():
        return _random_crop_or_pad(x, FIX_FRAMES)

    def use_slice():
        return x[:, start:start + FIX_FRAMES]

    x = tf.cond(start < 0, use_full, use_slice)
    x = _standardize(x)
    x = tf.expand_dims(x, -1)
    return x, y


def expand_to_windows(df: pd.DataFrame):
    paths_list  = []
    labels_list = []
    starts_list = []

    for _, row in df.iterrows():
        spec_path = str(row["spec_npy"])
        y = row[EMOTIONS].to_numpy(dtype=np.float32)

        if not os.path.isfile(spec_path):
            continue

        arr = np.load(spec_path).astype(np.float32, copy=False)
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.shape[0] != N_MELS:
            continue

        T = arr.shape[1]
        if T <= FIX_FRAMES:
            paths_list.append(spec_path)
            labels_list.append(y)
            starts_list.append(-1)
        else:
            for start in range(0, T - FIX_FRAMES + 1, STRIDE_FRAMES):
                paths_list.append(spec_path)
                labels_list.append(y)
                starts_list.append(start)

    paths  = np.array(paths_list, dtype=str)
    labels = np.stack(labels_list, axis=0) if labels_list else np.zeros((0, len(EMOTIONS)), dtype=np.float32)
    starts = np.array(starts_list, dtype=np.int32)
    return paths, labels, starts


# ============================================================
# Weighted KL loss (same as train_CRNN.py)
# ============================================================
def make_weighted_kl(class_w_np: np.ndarray):
    class_w = tf.constant(class_w_np.astype("float32"), dtype=tf.float32)

    def weighted_kl(y_true, y_pred):
        y_true = tf.clip_by_value(y_true, 1e-7, 1.0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        per_class = y_true * (tf.math.log(y_true) - tf.math.log(y_pred))
        weighted  = per_class * class_w
        denom     = tf.reduce_sum(class_w * y_true, axis=-1) + 1e-7
        return tf.reduce_sum(weighted, axis=-1) / denom

    return weighted_kl


# ============================================================
# Main training
# ============================================================
if __name__ == "__main__":
    # ----------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------
    if DATASET_TYPE == "main":
        map_df  = pd.read_csv(MAIN_MAPPING_CSV)[["spotifyid", "spec_npy"]]
        lab_df  = pd.read_csv(MAIN_LABELS_CSV)[["spotifyid"] + EMOTIONS]
        full_df = map_df.merge(lab_df, on="spotifyid", how="inner")
        full_df = full_df[
            full_df["spec_npy"].notna() &
            (full_df["spec_npy"].astype(str).str.strip() != "")
        ]
        full_df = full_df[full_df["spec_npy"].apply(lambda p: os.path.isfile(str(p)))]
        full_df = full_df.reset_index(drop=True)

    elif DATASET_TYPE == "deam":
        from DataSets.deam_processor import build_deam_dataframe
        full_df = build_deam_dataframe(
            annotations_csv=DEAM_ANNOTATIONS_CSV,
            audio_dir=DEAM_AUDIO_DIR,
            spec_output_dir=DEAM_SPEC_DIR,
        )
        full_df = full_df.reset_index(drop=True)

    else:
        raise ValueError(f"Unknown DATASET_TYPE '{DATASET_TYPE}'. Use 'main' or 'deam'.")

    if len(full_df) == 0:
        raise RuntimeError("No usable samples found. Check dataset paths and files.")

    # ----------------------------------------------------------
    # 2. Shuffle + train/val split (by song, not by window)
    # ----------------------------------------------------------
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val    = int(0.1 * len(full_df))
    train_df = full_df.iloc[n_val:].reset_index(drop=True)
    val_df   = full_df.iloc[:n_val].reset_index(drop=True)

    # Normalize labels to sum=1 (no-op for DEAM soft labels, needed for Spotify)
    train_df[EMOTIONS] = train_df[EMOTIONS].div(train_df[EMOTIONS].sum(axis=1), axis=0)
    val_df[EMOTIONS]   = val_df[EMOTIONS].div(val_df[EMOTIONS].sum(axis=1), axis=0)

    # ----------------------------------------------------------
    # 3. Class weights (inverse frequency on training songs)
    # ----------------------------------------------------------
    counts  = train_df[EMOTIONS].sum(axis=0).to_numpy(dtype=np.float32)
    inv     = 1.0 / (counts + 1e-8)
    class_w = np.clip(inv / inv.mean(), 0.25, 4.0)

    # ----------------------------------------------------------
    # 4. Window expansion + tf.data pipelines
    # ----------------------------------------------------------
    train_paths, train_y, train_starts = expand_to_windows(train_df)
    val_paths,   val_y,   val_starts   = expand_to_windows(val_df)

    print(f"Training songs: {len(train_df)}, windows: {len(train_paths)}")
    print(f"Validation songs: {len(val_df)}, windows: {len(val_paths)}")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_y, train_starts))
        .shuffle(2000)
        .map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_y, val_starts))
        .map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ----------------------------------------------------------
    # 5. Build + compile model
    # ----------------------------------------------------------
    from Model.model_zoo import get_model
    model   = get_model(MODEL_TYPE, n_mels=N_MELS, frames=FIX_FRAMES)
    loss_fn = make_weighted_kl(class_w)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=loss_fn,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="cat_acc"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )
    model.summary()

    # ----------------------------------------------------------
    # 6. Train — checkpoints use model-type prefix to avoid
    #    collision with the existing epoch_09.h5 used by
    #    PredictEmotionFromSong.py
    # ----------------------------------------------------------
    ckpt_pattern = f"Model/{MODEL_TYPE}_epoch_{{epoch:02d}}.h5"
    checkpoint_cb = ModelCheckpoint(
        filepath=ckpt_pattern,
        save_weights_only=False,
        save_freq="epoch",
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb],
    )

    # ----------------------------------------------------------
    # 7. Save final model
    # ----------------------------------------------------------
    final_path = f"Model/{MODEL_TYPE}_{DATASET_TYPE}_final.h5"
    model.save(final_path)
    print(f"Saved final model to: {final_path}")
    print("Class weights:", dict(zip(EMOTIONS, class_w.tolist())))
