from tensorflow import keras
from keras import layers

EMOTIONS = [
    "Wonder", "Transcendence", "Tenderness", "Nostalgia", "Peacefulness",
    "Power", "Joy", "Tension", "Sadness"
]


def build_crnn(n_mels: int = 128, frames: int = 512, n_classes: int = 9) -> keras.Model:
    """
    Bidirectional CRNN: 3 Conv2D blocks -> reshape -> 2 BiGRU -> GAP -> Dense.
    Identical architecture to train_CRNN.py's build_crnn().
    """
    inp = keras.Input(shape=(n_mels, frames, 1))

    x = layers.Conv2D(32, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)         # [m/2, t/2]

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)         # [m/4, t/4]

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)         # [m/8, t/8]

    # Time-major sequence for RNN
    x = layers.Permute((2, 1, 3))(x)                  # [batch, time, freq, ch]
    x = layers.Reshape((frames // 8, (n_mels // 8) * 128))(x)

    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inp, out)


def build_cnn(n_mels: int = 128, frames: int = 512, n_classes: int = 9) -> keras.Model:
    """
    CNN-only: 5 Conv2D blocks (32->64->128->256->512) -> GlobalAveragePooling2D -> Dense.

    Each block: Conv2D -> BatchNorm -> ReLU -> MaxPool2D(2,2).
    After 5 halvings: spatial dims are n_mels/32 x frames/32.
    Requires n_mels >= 32 and frames >= 32.

    Shape trace (n_mels=128, frames=512):
      (128, 512, 1) -> (64, 256, 32) -> (32, 128, 64) -> (16, 64, 128)
                    -> (8, 32, 256)  -> (4, 16, 512)
      GAP -> (512,) -> Dense(256) -> Dense(9)
    """
    assert n_mels >= 32, f"n_mels must be >= 32 for 5-block CNN, got {n_mels}"
    assert frames >= 32, f"frames must be >= 32 for 5-block CNN, got {frames}"

    inp = keras.Input(shape=(n_mels, frames, 1))
    x = inp
    for filters in [32, 64, 128, 256, 512]:
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inp, out)


def get_model(
    model_type: str,
    n_mels: int = 128,
    frames: int = 512,
    n_classes: int = 9,
) -> keras.Model:
    """
    Factory function for model architectures.

    Args:
        model_type: "crnn" | "cnn"
        n_mels: number of mel frequency bins
        frames: fixed time-frame window length
        n_classes: number of output emotion classes

    Returns:
        Uncompiled keras.Model
    """
    if model_type == "crnn":
        return build_crnn(n_mels, frames, n_classes)
    elif model_type == "cnn":
        return build_cnn(n_mels, frames, n_classes)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'crnn' or 'cnn'.")
