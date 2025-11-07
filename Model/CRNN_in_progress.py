import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------
# Config
# -------------------
N_MELS     = 128 
FIX_FRAMES = 512    # Taking crops across song and averaging eventually
BATCH_SIZE = 16
EPOCHS     = 20
EMOTIONS   = ["Wonder","Transcendence","Tenderness","Nostalgia","Peacefulness",
              "Power","Joy","Tension","Sadness"]

# -------------------
# Model (CRNN)
# -------------------
def build_crnn(n_mels=N_MELS, frames=FIX_FRAMES, n_classes=9):
    inp = keras.Input(shape=(n_mels, frames, 1))

    x = layers.Conv2D(32, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)              # [m/2, t/2]

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)              # [m/4, t/4]

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)              # [m/8, t/8]

    # Collapse freq into features and keep time for RNN
    B, H, W, C = None, n_mels//8, frames//8, 128
    x = layers.Permute((2,1,3))(x)                        # [batch, time, freq, ch]
    x = layers.Reshape((W, H*C))(x)                       # [batch, time, features]

    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)

    x = layers.GlobalAveragePooling1D()(x)                # pool over time
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.KLDivergence(),                # soft labels
        metrics=[
            keras.metrics.CategoricalAccuracy(name="cat_acc"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )
    return model


