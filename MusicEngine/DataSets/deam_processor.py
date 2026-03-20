"""
DEAM dataset processor: converts DEAM arousal/valence annotations into
GEMs-compatible soft labels and builds log-mel spectrogram .npy files.

User setup required:
  - Download DEAM audio to:       DataSets/DEAM/audio/  (files named <song_id>.mp3 or .wav)
  - Download annotations CSV to:  DataSets/DEAM/static_annotations_averaged_songs_1_2000.csv
    Available at: https://zenodo.org/records/1188976

The returned DataFrame has columns [song_id, spec_npy, Wonder, ..., Sadness]
and is directly compatible with the train.py pipeline.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

EMOTIONS = [
    "Wonder", "Transcendence", "Tenderness", "Nostalgia", "Peacefulness",
    "Power", "Joy", "Tension", "Sadness"
]

# ============================================================
# GEMs reference coordinates in normalized VA space [0, 1].
# Valence on X-axis, Arousal on Y-axis.
# Based on Russell (1980) Circumplex Model of Affect and
# empirical GEMs coordinates from Zentner, Grandjean & Scherer (2008).
# ============================================================
GEM_COORDS: dict[str, tuple[float, float]] = {
    #                  (valence, arousal)
    "Wonder":          (0.75, 0.75),
    "Transcendence":   (0.80, 0.55),
    "Tenderness":      (0.70, 0.30),
    "Nostalgia":       (0.55, 0.35),
    "Peacefulness":    (0.65, 0.20),
    "Power":           (0.50, 0.85),
    "Joy":             (0.85, 0.80),
    "Tension":         (0.20, 0.80),
    "Sadness":         (0.20, 0.25),
}

DEAM_SOFTMAX_TEMP = 3.0  # higher = sharper / more peaked soft labels


def va_to_gems_soft_label(
    mean_arousal: float,
    mean_valence: float,
    temp: float = DEAM_SOFTMAX_TEMP,
) -> np.ndarray:
    """
    Convert DEAM arousal/valence (1–9 scale) to a 9-dim GEMs soft label.

    Steps:
      1. Normalize to [0, 1]:  norm = (score - 1) / 8.0
      2. Compute Euclidean distance from (v_norm, a_norm) to each GEM coord.
      3. Soft-assign via temperature-scaled exponential:
           weight[i] = exp(-temp * dist[i])
           label = weight / sum(weight)

    Args:
        mean_arousal: float in [1, 9]
        mean_valence: float in [1, 9]
        temp: sharpness temperature (default 3.0)

    Returns:
        np.ndarray of shape (9,) summing to 1.0
    """
    a_norm = (mean_arousal - 1.0) / 8.0
    v_norm = (mean_valence - 1.0) / 8.0

    weights = []
    for emo in EMOTIONS:
        v_ref, a_ref = GEM_COORDS[emo]
        dist = np.sqrt((v_norm - v_ref) ** 2 + (a_norm - a_ref) ** 2)
        weights.append(np.exp(-temp * dist))

    weights = np.array(weights, dtype=np.float32)
    return weights / (weights.sum() + 1e-8)


def build_deam_dataframe(
    annotations_csv: str,
    audio_dir: str,
    spec_output_dir: str,
    sr: int = 32000,
    n_fft: int = 1024,
    hop: int = 320,
    n_mels: int = 128,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Process DEAM audio files into log-mel spectrogram .npy files and return
    a DataFrame compatible with the train.py pipeline.

    Args:
        annotations_csv:  path to DEAM static annotations CSV.
                          Required columns: song_id, mean_arousal, mean_valence.
        audio_dir:        directory containing DEAM audio files (*.mp3 or *.wav),
                          named <song_id>.mp3 / <song_id>.wav.
        spec_output_dir:  where .npy spectrogram files are saved.
        sr, n_fft, hop, n_mels: spectrogram params — must match training config.
        skip_existing:    if True, reuse already-computed .npy files.

    Returns:
        pd.DataFrame with columns [song_id, spec_npy, Wonder, ..., Sadness].
        Only rows where audio was found and successfully processed are included.
    """
    # Import here to avoid a hard failure if librosa is unavailable at module load
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from SpotifyToSpectrogram.mp3ToSpectrogram import audio_to_logmel_array

    ann_df = pd.read_csv(annotations_csv)
    ann_df.columns = ann_df.columns.str.strip()  # DEAM CSVs have leading spaces

    required_cols = {"song_id", "mean_arousal", "mean_valence"}
    if not required_cols.issubset(ann_df.columns):
        raise ValueError(
            f"annotations_csv missing columns. "
            f"Expected {required_cols}, got {set(ann_df.columns)}"
        )

    Path(spec_output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in ann_df.iterrows():
        song_id = int(row["song_id"])
        mean_a = float(row["mean_arousal"])
        mean_v = float(row["mean_valence"])

        # Skip rows with out-of-range VA values
        if not (1.0 <= mean_a <= 9.0 and 1.0 <= mean_v <= 9.0):
            continue

        # Locate audio file — try .mp3 first, then .wav
        audio_path: Optional[str] = None
        for ext in [".mp3", ".wav"]:
            candidate = os.path.join(audio_dir, f"{song_id}{ext}")
            if os.path.isfile(candidate):
                audio_path = candidate
                break

        if audio_path is None:
            continue  # audio not downloaded; skip silently

        npy_path = os.path.join(spec_output_dir, f"deam_{song_id}.npy")

        if not (skip_existing and os.path.isfile(npy_path)):
            try:
                spec = audio_to_logmel_array(audio_path, sr=sr, n_fft=n_fft,
                                             hop=hop, n_mels=n_mels)
                np.save(npy_path, spec)
            except Exception as e:
                print(f"[DEAM] Failed to process song_id={song_id}: {e}")
                continue

        soft_label = va_to_gems_soft_label(mean_a, mean_v)

        entry: dict = {"song_id": song_id, "spec_npy": npy_path}
        for i, emo in enumerate(EMOTIONS):
            entry[emo] = float(soft_label[i])
        rows.append(entry)

    df = pd.DataFrame(rows)
    print(f"[DEAM] Built dataframe: {len(df)} songs processed out of {len(ann_df)} annotations.")
    return df
