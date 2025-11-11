import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

# =========================
# Config
# =========================
N_MELS     = 128
FIX_FRAMES = 512        # crop/pad to this many frames
BATCH_SIZE = 16
EPOCHS     = 5          # keep small for the single-row test
JOIN_KEY   = "spotifyid"
EMOTIONS   = ["Wonder","Transcendence","Tenderness","Nostalgia","Peacefulness",
              "Power","Joy","Tension","Sadness"]

# Paths (edit if needed)
MAPPING_CSV = "DataSets/SpotifyIDMappings.csv"       # must contain [spotifyid, spec_npy]
LABELS_CSV  = "DataSets/SpotifyMetaToGems_Final.csv" # must contain [spotifyid, the 9 EMOTIONS columns]


# =========================
# Preprocessing helpers
# =========================
def _random_crop_or_pad(x: tf.Tensor, target_frames: int = FIX_FRAMES) -> tf.Tensor:
    # x: [mels, time]
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
# I/O
# =========================
def load_spectrogram_npy(path: str) -> np.ndarray:
    """Return float32 array [n_mels, time]."""
    arr = np.load(path)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    assert arr.shape[0] == N_MELS, f"Expected {N_MELS} mels, got {arr.shape[0]}"
    return arr

def load_one_example(mapping_csv: str,
                     labels_csv: str,
                     join_key: str = JOIN_KEY,
                     row_index: int = 0,
                     or_spotifyid: str | None = None):
    """
    Loads exactly one (X, y) pair by row index or spotifyid.

    Returns:
      X: np.ndarray with shape [1, N_MELS, FIX_FRAMES, 1]
      y: np.ndarray with shape [1, 9]
    """
    map_df = pd.read_csv(mapping_csv)
    lab_df = pd.read_csv(labels_csv)

    # Ensure emotion columns exist; if labels CSV has extras, select only EMOTIONS
    missing = [c for c in EMOTIONS if c not in lab_df.columns]
    if missing:
        raise ValueError(f"Labels CSV is missing emotion columns: {missing}")

    # Select the specific row
    if or_spotifyid is not None:
        map_row = map_df.loc[map_df[join_key] == or_spotifyid]
        lab_row = lab_df.loc[lab_df[join_key] == or_spotifyid]
        if map_row.empty or lab_row.empty:
            raise ValueError(f"spotifyid={or_spotifyid} not found in both CSVs.")
        map_row = map_row.iloc[0]
        lab_row = lab_row.iloc[0]
    else:
        map_row = map_df.iloc[row_index]
        # find matching labels row by JOIN_KEY
        skey = map_row[join_key]
        match = lab_df.loc[lab_df[join_key] == skey]
        if match.empty:
            raise ValueError(f"No labels row found for {join_key}={skey}")
        lab_row = match.iloc[0]

    spec_path = str(map_row["spec_npy"])
    if not os.path.isfile(spec_path):
        raise FileNotFoundError(f"Spec path not found: {spec_path}")

    # Load and preprocess spectrogram
    spec = load_spectrogram_npy(spec_path)     # [mels, time]
    x = tf.convert_to_tensor(spec)             # tf.Tensor
    x = _random_crop_or_pad(x, FIX_FRAMES)     # [mels, FIX_FRAMES]
    x = _standardize(x)
    x = tf.expand_dims(x, -1)                  # [mels, FIX_FRAMES, 1]
    x = tf.expand_dims(x, 0)                   # [1, mels, FIX_FRAMES, 1]

    # Labels (ensure sum≈1)
    y = lab_row[EMOTIONS].to_numpy(dtype=np.float32)  # [9]
    y_sum = y.sum()
    if not np.isclose(y_sum, 1.0, atol=1e-3):
        # Normalize if slightly off
        y = y / (y_sum + 1e-8)
    y = y[None, :]                                # [1, 9]

    return x.numpy(), y


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
    x = layers.Reshape((FIX_FRAMES // 8, (n_mels // 8) * 128))(x)

    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.KLDivergence(),  # soft labels
        metrics=[
            keras.metrics.CategoricalAccuracy(name="cat_acc"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )
    return model


# =========================
# Single-row training
# =========================
if __name__ == "__main__":
    # --- Load one example by row index (or set or_spotifyid="...") ---
    # X1, y1 = load_one_example(
    #     mapping_csv=MAPPING_CSV,
    #     labels_csv=LABELS_CSV,
    #     join_key=JOIN_KEY,
    #     row_index=0,
    #     or_spotifyid=None,          # or: "some_spotify_id"
    # )
    # # X1: [1, N_MELS, FIX_FRAMES, 1]; y1: [1, 9]

    # model = build_crnn()
    # model.summary()

    # # Fit on the single example. This is just to verify end-to-end plumbing.
    # # (You can also call model.train_on_batch(X1, y1).)
    # model.fit(
    #     X1, y1,
    #     epochs=EPOCHS,
    #     batch_size=1,
    #     verbose=1
    # )

    # # Quick prediction on the same clip
    # preds = model.predict(X1, verbose=0)[0]
    # print({emo: float(p) for emo, p in zip(EMOTIONS, preds)})


    # =========================
    # (Later) Full-dataset pipeline — uncomment when you have all rows
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
    
    def _map_fn(path, y):
        x = load_npy_tensor(path)                 # [mels, time]
        x = _random_crop_or_pad(x, FIX_FRAMES)    # [mels, FIX_FRAMES]
        x = _standardize(x)
        x = tf.expand_dims(x, -1)                 # [mels, frames, 1]
        return x, y
    
    # Build a joined DataFrame with paths + labels
    map_df = pd.read_csv(MAPPING_CSV)[[JOIN_KEY, "spec_npy"]]
    lab_df = pd.read_csv(LABELS_CSV)[[JOIN_KEY] + EMOTIONS]
    full_df = map_df.merge(lab_df, on=JOIN_KEY, how="inner")
    full_df = full_df[full_df["spec_npy"].notna() & (full_df["spec_npy"].str.strip() != "")]
    
    # Train/val split
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = int(0.1 * len(full_df))
    train_df = full_df.iloc[n_val:].reset_index(drop=True)
    val_df   = full_df.iloc[:n_val].reset_index(drop=True)
    
    # tf.data pipelines
    train_paths = train_df["spec_npy"].astype(str).values
    train_y     = train_df[EMOTIONS].astype(np.float32).values
    val_paths   = val_df["spec_npy"].astype(str).values
    val_y       = val_df[EMOTIONS].astype(np.float32).values
    
    # Optional: re-normalize rows to sum=1 if CSV rounding drifted
    train_y = train_y / (train_y.sum(axis=1, keepdims=True) + 1e-8)
    val_y   = val_y / (val_y.sum(axis=1, keepdims=True) + 1e-8)
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
    train_ds = train_ds.shuffle(1000).map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y))
    val_ds = val_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    model = build_crnn()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    SAVE_DIR = "Model"
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_path = os.path.join(SAVE_DIR, "crnn_emotion_model.h5")
    model.save(model_path)
