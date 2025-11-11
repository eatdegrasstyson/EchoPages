import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from SpotifyToSpectrogram.mp3ToSpectrogram import audio_to_logmel_array
from SpotifyToSpectrogram.name_to_audio import download_mp3_from_spotify_id
from SpotifyToSpectrogram.get_metadata import get_data_from_id

MODEL_PATH = "Model/crnn_emotion_model.h5"
TMP_AUDIO_PATH = Path("Temp/audio")
TMP_SPEC_PATH  = Path("Temp/spec")
N_MELS = 128
FIX_FRAMES = 512
EMOTIONS = ["Wonder","Transcendence","Tenderness","Nostalgia","Peacefulness",
            "Power","Joy","Tension","Sadness"]

TMP_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
TMP_SPEC_PATH.mkdir(parents=True, exist_ok=True)


#Same preprocessing as training
def _random_crop_or_pad(x: tf.Tensor, target_frames: int = FIX_FRAMES) -> tf.Tensor:
    t = tf.shape(x)[1]
    def pad():
        pad_amt = target_frames - t
        return tf.pad(x, [[0, 0], [0, pad_amt]])
    def crop():
        start = tf.maximum(0, tf.random.uniform((), 0, t - target_frames + 1, dtype=tf.int32))
        return x[:, start:start + target_frames]
    return tf.cond(t < target_frames, pad, crop)

def _standardize(x: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(x)
    std  = tf.math.reduce_std(x) + 1e-6
    return (x - mean) / std


def predict_emotion(spotify_id):
    #Download from YouTube
    track = get_data_from_id(spotify_id)
    mp3_path = download_mp3_from_spotify_id(track, TMP_AUDIO_PATH)

    if not mp3_path or not os.path.exists(mp3_path):
        print("Failed to download or locate audio.")
        return

    #Convert to spectrogram
    spec = audio_to_logmel_array(mp3_path)
    np.save(TMP_SPEC_PATH / f"{Path(mp3_path).stem}.npy", spec)

    #Convert to tensor and preprocess
    x = tf.convert_to_tensor(spec, dtype=tf.float32)
    x = _random_crop_or_pad(x, FIX_FRAMES)
    x = _standardize(x)
    x = tf.expand_dims(x, -1)  #[n_mels, frames, 1]
    x = tf.expand_dims(x, 0)   #[1, n_mels, frames, 1]

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)

    # Predict
    print("Predicting emotion...")
    preds = model.predict(x, verbose=0)[0]

    # Show results sorted by confidence
    results = sorted(zip(EMOTIONS, preds), key=lambda kv: kv[1], reverse=True)
    print("\n=== Emotion Prediction ===")
    for emo, p in results:
        print(f"{emo:15s}: {p:.3f}")

    top_emo, top_p = results[0]
    print(f"\nDominant emotion: {top_emo} ({top_p:.2%})")


# =========================
# CLI entry
# =========================
if __name__ == "__main__":

    spotifyID = sys.argv[1]
    predict_emotion(spotifyID)
