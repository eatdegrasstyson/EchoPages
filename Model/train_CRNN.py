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

if __name__ == "__main__":
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
