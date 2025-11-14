import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

# =========================
# Config
# =========================
N_MELS       = 128
FIX_FRAMES   = 512          # window length in frames
STRIDE_FRAMES = FIX_FRAMES // 2   # 50% overlap between windows
BATCH_SIZE   = 16
EPOCHS       = 5
JOIN_KEY     = "spotifyid"
EMOTIONS     = [
    "Wonder","Transcendence","Tenderness","Nostalgia","Peacefulness",
    "Power","Joy","Tension","Sadness"
]

# Paths
MAPPING_CSV = "DataSets/SpotifyIDMappings.csv"       # must contain [spotifyid, spec_npy]
LABELS_CSV  = "DataSets/SpotifyMetaToGems_Final.csv" # must contain [spotifyid] + EMOTIONS

# =========================
# Preprocessing helpers
# =========================
def _random_crop_or_pad(x: tf.Tensor, target_frames: int = FIX_FRAMES) -> tf.Tensor:
    """If time < target_frames: pad; else random crop."""
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
# Model
# =========================
def build_crnn(n_mels=N_MELS, frames=FIX_FRAMES, n_classes=9) -> keras.Model:
    inp = keras.Input(shape=(n_mels, frames, 1))
    x = layers.Conv2D(32, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)     # [m/2, t/2]

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)     # [m/4, t/4]

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)     # [m/8, t/8]

    # time-major sequence for RNN
    x = layers.Permute((2, 1, 3))(x)              # [batch, time, freq, ch]
    x = layers.Reshape((frames // 8, (n_mels // 8) * 128))(x)

    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    return model

# =========================
# I/O + window expansion
# =========================
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
    """
    path: scalar string tensor to .npy
    y   : [9] soft label
    start: scalar int32; -1 means "use full clip with crop/pad"
    """
    x = load_npy_tensor(path)                 # [mels, time]
    t = tf.shape(x)[1]

    def use_full():
        return _random_crop_or_pad(x, FIX_FRAMES)

    def use_slice():
        # deterministic slice [start:start+FIX_FRAMES]
        return x[:, start:start + FIX_FRAMES]

    x = tf.cond(start < 0, use_full, use_slice)
    x = _standardize(x)
    x = tf.expand_dims(x, -1)                 # [mels, frames, 1]
    return x, y

def expand_to_windows(df: pd.DataFrame):
    """
    For each song in df, expand into multiple windows over the full spectrogram.

    Returns:
      paths:  np.ndarray of str paths (one per window)
      labels: np.ndarray of shape [num_windows, 9]
      starts: np.ndarray of int32 start indices (frame idx; -1 means full-clip mode)
    """
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
            # skip malformed spectrograms
            continue

        T = arr.shape[1]

        if T <= FIX_FRAMES:
            # short track: single "full clip" window; handled with pad/crop later
            paths_list.append(spec_path)
            labels_list.append(y)
            starts_list.append(-1)  # sentinel
        else:
            # sliding windows across the whole song
            for start in range(0, T - FIX_FRAMES + 1, STRIDE_FRAMES):
                paths_list.append(spec_path)
                labels_list.append(y)
                starts_list.append(start)

    paths  = np.array(paths_list, dtype=str)
    labels = np.stack(labels_list, axis=0) if labels_list else np.zeros((0, len(EMOTIONS)), dtype=np.float32)
    starts = np.array(starts_list, dtype=np.int32)

    return paths, labels, starts

# =========================
# Weighted KL loss
# =========================
def make_weighted_kl(class_w_np: np.ndarray):
    """
    class_w_np: np.ndarray shape (9,), higher weight => larger contribution for that emotion.
    Uses per-sample normalization so overall loss scale is stable regardless of label mixture.
    """
    class_w = tf.constant(class_w_np.astype("float32"), dtype=tf.float32)  # (9,)

    def weighted_kl(y_true, y_pred):
        y_true = tf.clip_by_value(y_true, 1e-7, 1.0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        # KL per class: y * (log y - log p)
        per_class = y_true * (tf.math.log(y_true) - tf.math.log(y_pred))  # [B,9]
        weighted = per_class * class_w                                    # [B,9]
        denom = tf.reduce_sum(class_w * y_true, axis=-1) + 1e-7           # [B]
        return tf.reduce_sum(weighted, axis=-1) / denom                   # [B]

    return weighted_kl

# =========================
# Main training
# =========================
if __name__ == "__main__":
    # Join mapping + labels
    map_df = pd.read_csv(MAPPING_CSV)[[JOIN_KEY, "spec_npy"]]
    lab_df = pd.read_csv(LABELS_CSV)[[JOIN_KEY] + EMOTIONS]
    full_df = map_df.merge(lab_df, on=JOIN_KEY, how="inner")

    # Drop rows with empty/missing spectrogram paths and non-existent files
    full_df = full_df[full_df["spec_npy"].notna() & (full_df["spec_npy"].astype(str).str.strip() != "")]
    full_df = full_df[full_df["spec_npy"].apply(lambda p: os.path.isfile(str(p)))]
    full_df = full_df.reset_index(drop=True)

    # Shuffle and split by SONG (not by window)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = int(0.1 * len(full_df))
    train_df = full_df.iloc[n_val:].reset_index(drop=True)
    val_df   = full_df.iloc[:n_val].reset_index(drop=True)

    # (Optional) ensure labels rows sum to 1 at song level
    train_df[EMOTIONS] = train_df[EMOTIONS].div(train_df[EMOTIONS].sum(axis=1), axis=0)
    val_df[EMOTIONS]   = val_df[EMOTIONS].div(val_df[EMOTIONS].sum(axis=1), axis=0)

    # Compute class weights from training songs (not windows)
    counts = train_df[EMOTIONS].sum(axis=0).to_numpy(dtype=np.float32)  # [9]
    eps = 1e-8
    inv = 1.0 / (counts + eps)
    class_w = inv / inv.mean()
    class_w = np.clip(class_w, 0.25, 4.0)  # avoid extreme weights

    # Expand to windows (each window shares its song's label)
    train_paths, train_y_win, train_starts = expand_to_windows(train_df)
    val_paths,   val_y_win,   val_starts   = expand_to_windows(val_df)

    print(f"Num training songs: {len(train_df)}, windows: {len(train_paths)}")
    print(f"Num validation songs: {len(val_df)}, windows: {len(val_paths)}")

    # Build tf.data pipelines
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y_win, train_starts))
    train_ds = train_ds.shuffle(2000).map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y_win, val_starts))
    val_ds = val_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build & compile model with weighted KL
    model = build_crnn()
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
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Save model (you can switch to JSON+weights if you want clean inference loading)
    SAVE_DIR = "Model"
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "crnn_emotion_model_weighted_windows.h5")
    model.save(model_path)
    print("Saved trained model to:", model_path)
    print("Class weights:", dict(zip(EMOTIONS, class_w.tolist())))
