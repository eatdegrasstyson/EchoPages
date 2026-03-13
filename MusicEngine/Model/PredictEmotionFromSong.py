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
STRIDE_FRAMES = FIX_FRAMES // 2   # match training overlap (50%)
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
    if not os.path.isfile(arch_path):
        raise FileNotFoundError(f"Architecture JSON not found at {arch_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    with open(arch_path) as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights(weights_path)
    return model

def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred) * 0.0

def load_saved_model(path: str) -> keras.Model:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    
    # Load the model with a placeholder for the custom weighted KL loss
    return keras.models.load_model(
        path,
        custom_objects={"weighted_kl": dummy_loss}  # match name used in training
    )


# =========================
# Full-song multi-window prediction
# =========================
def predict_emotion_full_song(model: keras.Model,
                              spec: np.ndarray,
                              fix_frames: int = FIX_FRAMES,
                              stride_frames: int = STRIDE_FRAMES,
                              chunking: bool = False) -> tuple[np.ndarray,np.ndarray]:
    """
    spec: np.ndarray [n_mels, time]
    Returns: np.ndarray [9] averaged emotion distribution over the whole song.

    Matches training logic: slide fixed windows across entire song and treat
    each window as a training example. At inference we average window predictions.
    """
    if spec.ndim != 2 or spec.shape[0] != N_MELS:
        raise ValueError(f"Expected spectrogram shape [N_MELS, time], got {spec.shape}")

    T = spec.shape[1]

    # If the song is shorter than one window, just pad/crop once.
    if T <= fix_frames:
        x = tf.convert_to_tensor(spec, dtype=tf.float32)
        x = _random_crop_or_pad(x, fix_frames)   # pad/crop to FIX_FRAMES
        x = _standardize(x)
        x = tf.expand_dims(x, -1)                # [mels, frames, 1]
        x = tf.expand_dims(x, 0)                 # [1, mels, frames, 1]
        return model.predict(x, verbose=0)[0]

    # Build sliding window start indices
    starts = list(range(0, T - fix_frames + 1, stride_frames))
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
    
    # assumes: hop_size is same as stride_frames in frames
    # and spectrogram was created at sr=32000 Hz
    sr = 32000
    hop_size = 320  # must match audio_to_logmel_array hop
    chunk_times = np.array(starts) * hop_size / sr


    if not chunking:
        return preds.mean(axis=0),chunk_times                 # [9], averaged distribution
    else:
        return preds,chunk_times


# =========================
# End-to-end prediction
# =========================
def predict_emotion(spotify_id: str):
    # 1. Download audio via YouTube based on Spotify metadata
    track = get_data_from_id(spotify_id)
    mp3_path = download_mp3_from_spotify_id(track, TMP_AUDIO_PATH)

    if not mp3_path or not os.path.exists(mp3_path):
        print("Failed to download or locate audio.")
        return

    # 2. Convert audio to log-mel spectrogram
    spec = audio_to_logmel_array(mp3_path)  # [N_MELS, time]
    npy_path = TMP_SPEC_PATH / f"{Path(mp3_path).stem}.npy"
    np.save(npy_path, spec)

    # 3. Load model
    print("Loading model...")
    model = load_saved_model("Model/epoch_09.h5")

    # 4. Predict over full song via sliding windows + averaging
    print("Predicting emotion over full song (sliding windows)...")
    preds,_ = predict_emotion_full_song(model, spec)

    # 5. Show results
    results = sorted(zip(EMOTIONS, preds), key=lambda kv: kv[1], reverse=True)
    print("\n=== Emotion Prediction ===")
    for emo, p in results:
        print(f"{emo:15s}: {p:.3f}")

    top_emo, top_p = results[0]
    print(f"\nDominant emotion: {top_emo} ({top_p:.2%})")


def predict_spectrogram(spec, chunking=False):
    # 3. Load model
    print("Loading model...")
    model = load_saved_model("Model/epoch_09.h5")
    preds,times = predict_emotion_full_song(model, spec, chunking=chunking)
    return preds,times

def predict_multiple_songs(spec_paths, chunking=True):
    """
    Predicts multiple songs using the same loaded model, loading it only once.
    
    spec_paths: list of npy spectrogram paths (relative to root)
    chunking: whether to run chunked predictions

    Returns:
        list of tuples: (spec_path, avg_vec, chunks)
    """
    # Load model once
    model = load_saved_model("Model/epoch_09.h5")

    results = []
    for spec_path in spec_paths:
        try:
            spec = np.load(spec_path)
        except Exception as e:
            print(f"Skipping {spec_path}: failed to load spectrogram ({e})")
            continue
        preds, times = predict_emotion_full_song(model, spec, chunking=chunking)

        avg_vec = preds.mean(axis=0)
        chunks = chunkingData_dp(preds, times) if chunking else []

        results.append((spec_path, avg_vec, chunks))

    return results

def cosine_dist(a, b, eps=1e-8):
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

def chunkingData(preds, times):
    hop_size = 320
    sr = 32000
    WINDOW_SEC = FIX_FRAMES * hop_size / sr

    #Dynamic params:

    SMOOTH_WINDOW = 5
    COMPARE_WINDOW = 6

    #Smoothed predictions
    smoothed = []
    for i in range(len(preds)):
        start = max(0, i - SMOOTH_WINDOW)
        end = min(len(preds), i + SMOOTH_WINDOW + 1)
        smoothed.append(np.mean(preds[start:end], axis=0))
    preds = np.array(smoothed)

    structural_dists = []

    for i in range(COMPARE_WINDOW, len(preds) - COMPARE_WINDOW):
        past_mean = np.mean(preds[i-COMPARE_WINDOW:i], axis=0)
        future_mean = np.mean(preds[i:i+COMPARE_WINDOW], axis=0)

        v = cosine_dist(past_mean, future_mean)
        structural_dists.append(v)

    structural_dists = np.array(structural_dists)

    #Reduce num for more chunks, increase for less chunks
    chunkMultiplier = 85
    SPLIT_THRESHOLD = np.percentile(structural_dists, chunkMultiplier)

    MIN_SECONDS = 18.0
    MERGE_THRESHOLD = 0.12

    dt = times[1] - times[0]
    min_frames = int(MIN_SECONDS / dt)

    # Detect structural change via past vs future window comparison
    candidate_splits = []
    for i in range(COMPARE_WINDOW, len(preds) - COMPARE_WINDOW):
        past_mean = np.mean(preds[i-COMPARE_WINDOW:i], axis=0)
        future_mean = np.mean(preds[i:i+COMPARE_WINDOW], axis=0)

        v = cosine_dist(past_mean, future_mean)

        if v > SPLIT_THRESHOLD:
            candidate_splits.append(i)

    # Enforce min distance between splits
    filtered_splits = []
    last_split = -9999
    for s in candidate_splits:
        if s - last_split >= min_frames:
            filtered_splits.append(s)
            last_split = s

    # Build chunks from filtered splits
    chunks = []
    split_points = [0] + filtered_splits + [len(preds)]

    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i+1]

        mean_vec = np.mean(preds[start_idx:end_idx], axis=0)
        top3_indices = mean_vec.argsort()[-9:][::-1]

        chunks.append({
            "start": times[start_idx],
            "end": times[end_idx-1] + WINDOW_SEC,
            "emotion": mean_vec,
            "label": [(EMOTIONS[j], mean_vec[j]) for j in top3_indices]
        })

    # Merge emotionally similar adjacent chunks
    merged = [chunks[0]]
    for c in chunks[1:]:
        prev = merged[-1]
        if cosine_dist(prev["emotion"], c["emotion"]) < MERGE_THRESHOLD:
            prev["end"] = c["end"]
            prev["emotion"] = (prev["emotion"] + c["emotion"]) / 2
        else:
            merged.append(c)

    for idx, c in enumerate(merged):
        dur = c["end"] - c["start"]
        top3_str = ", ".join([f"{e} ({v:.2f})" for e, v in c['label']])
        print(f"[{idx}] {c['start']:6.1f}s → {c['end']:6.1f}s ({dur:5.1f}s) | {top3_str}")

    return merged


#Claud stuff
def _print_raw_windows(preds, times, window_sec):
    print("\n=== Raw Windows ===")
    for i, vec in enumerate(preds):
        top3_idx = vec.argsort()[-3:][::-1].astype(int)
        top3_emotions = [EMOTIONS[j] for j in top3_idx]
        start_time = times[i]
        end_time = start_time + window_sec
        print(f"[{i}] {start_time:6.1f}s → {end_time:6.1f}s ({window_sec:5.1f}s) | {', '.join(top3_emotions)}")

def _build_chunk(preds, times, start_idx, end_idx, window_sec):
    mean_vec = np.mean(preds[start_idx:end_idx], axis=0)
    top_indices = mean_vec.argsort()[::-1]  # all indices sorted strongest → weakest
    return {
        "start": times[start_idx],
        "end": times[end_idx - 1] + window_sec,
        "emotion": mean_vec,
        "label": [(EMOTIONS[j], mean_vec[j]) for j in top_indices]
    }


def _print_chunks(chunks, header="Emotion Chunks"):
    print(f"\n=== {header} ===")
    for idx, c in enumerate(chunks):
        dur = c["end"] - c["start"]
        top3_str = ", ".join([f"{e} ({v:.2f})" for e, v in c['label']])
        print(f"[{idx}] {c['start']:6.1f}s → {c['end']:6.1f}s ({dur:5.1f}s) | {top3_str}")

def chunkingData_stable_merge(preds, times,
                              run_threshold=0.25,
                              merge_threshold=0.12,
                              ema_alpha=0.3):
    """
    Phase 1 — Stable runs: extend a run as long as every window has
              cosine similarity >= run_threshold to the run centroid.
    Phase 2 — Merge: merge adjacent micro-runs whose centroids are
              within merge_threshold cosine distance.

    EMA smoothing (alpha) is applied to preds before chunking to
    absorb window-to-window noise.
    """
    hop_size = 320
    sr = 32000
    window_sec = FIX_FRAMES * hop_size / sr

    #_print_raw_windows(preds, times, window_sec)

    n = len(preds)

    # --- EMA smoothing ---
    smoothed = np.empty_like(preds)
    smoothed[0] = preds[0]
    for i in range(1, n):
        smoothed[i] = ema_alpha * preds[i] + (1 - ema_alpha) * smoothed[i - 1]

    # --- Phase 1: build maximally-stable micro-runs ---
    runs = []  # list of (start_idx, end_idx) exclusive end
    run_start = 0
    centroid = smoothed[0].copy()
    count = 1

    for i in range(1, n):
        dist = cosine_dist(smoothed[i], centroid)
        if dist < run_threshold:
            # extend run, update centroid incrementally
            centroid = (centroid * count + smoothed[i]) / (count + 1)
            count += 1
        else:
            runs.append((run_start, i))
            run_start = i
            centroid = smoothed[i].copy()
            count = 1
    runs.append((run_start, n))

    # --- Phase 2: merge adjacent runs with similar centroids ---
    def run_centroid(start, end):
        return np.mean(smoothed[start:end], axis=0)

    merged = [runs[0]]
    for run in runs[1:]:
        prev_start, prev_end = merged[-1]
        cur_start, cur_end = run
        prev_c = run_centroid(prev_start, prev_end)
        cur_c = run_centroid(cur_start, cur_end)
        if cosine_dist(prev_c, cur_c) < merge_threshold:
            merged[-1] = (prev_start, cur_end)
        else:
            merged.append(run)

    # Repeat merge passes until stable
    changed = True
    while changed:
        changed = False
        new_merged = [merged[0]]
        for run in merged[1:]:
            prev_start, prev_end = new_merged[-1]
            cur_start, cur_end = run
            prev_c = run_centroid(prev_start, prev_end)
            cur_c = run_centroid(cur_start, cur_end)
            if cosine_dist(prev_c, cur_c) < merge_threshold:
                new_merged[-1] = (prev_start, cur_end)
                changed = True
            else:
                new_merged.append(run)
        merged = new_merged

    # --- Build output chunks (use original preds for emotion vectors) ---
    chunks = []
    for start, end in merged:
        chunks.append(_build_chunk(preds, times, start, end, window_sec))

    _print_chunks(chunks, header="Emotion Chunks (Stable-Merge)")
    return chunks


#Segmented Least Squares via DP
#0.6 needed to get good for hotel cali, 0.8 for all songs tested

def chunkingData_dp(preds, times,
                    lam=0.8,
                    ema_alpha=0.3):
    """
    Globally optimal segmentation via dynamic programming.

    Minimizes:  sum_chunks( intra-chunk variance ) + lambda * num_chunks

    where intra-chunk variance = sum of squared cosine distances from each
    window to the chunk mean.

    lambda (lam) controls the trade-off: higher = fewer, larger chunks.
    """
    hop_size = 320
    sr = 32000
    window_sec = FIX_FRAMES * hop_size / sr

    #_print_raw_windows(preds, times, window_sec)

    n = len(preds)

    # --- EMA smoothing ---
    smoothed = np.empty_like(preds)
    smoothed[0] = preds[0]
    for i in range(1, n):
        smoothed[i] = ema_alpha * preds[i] + (1 - ema_alpha) * smoothed[i - 1]

    # --- Precompute segment costs ---
    # cost[i][j] = intra-chunk variance for windows [i, j)
    # To avoid O(n^3), compute incrementally using prefix sums.
    # Variance = sum_k ||v_k - mean||^2  (cosine distance squared)
    # We approximate with Euclidean variance for DP efficiency, which is
    # monotonically related to cosine distance for normalized-ish vectors.

    # prefix_sum[i] = sum of smoothed[0..i-1]
    prefix_sum = np.zeros((n + 1, smoothed.shape[1]))
    prefix_sq_sum = np.zeros(n + 1)  # sum of ||v_k||^2
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + smoothed[i]
        prefix_sq_sum[i + 1] = prefix_sq_sum[i] + np.dot(smoothed[i], smoothed[i])

    def segment_cost(i, j):
        """Variance of smoothed[i:j] = sum||v - mean||^2."""
        length = j - i
        if length <= 1:
            return 0.0
        seg_sum = prefix_sum[j] - prefix_sum[i]
        seg_sq = prefix_sq_sum[j] - prefix_sq_sum[i]
        mean_sq_norm = np.dot(seg_sum, seg_sum) / (length * length)
        return seg_sq - length * mean_sq_norm

    # --- DP ---
    # dp[j] = minimum cost to segment windows [0, j)
    dp = np.full(n + 1, np.inf)
    dp[0] = 0.0
    parent = np.zeros(n + 1, dtype=int)  # parent[j] = optimal split point i

    for j in range(1, n + 1):
        for i in range(j):
            cost = dp[i] + segment_cost(i, j) + lam
            if cost < dp[j]:
                dp[j] = cost
                parent[j] = i

    # --- Backtrack to recover chunk boundaries ---
    boundaries = []
    j = n
    while j > 0:
        i = parent[j]
        boundaries.append((i, j))
        j = i
    boundaries.reverse()

    # --- Build output chunks (use original preds for emotion vectors) ---
    chunks = []
    for start, end in boundaries:
        chunks.append(_build_chunk(preds, times, start, end, window_sec))

    _print_chunks(chunks, header="Emotion Chunks (DP Segmented)")
    return chunks

if __name__ == "__main__":
    '''
    if len(sys.argv) < 2:
        print("Usage: python -m YourModule.predict <spotify_id>")
        sys.exit(1)

    spotify_id = sys.argv[1]
    predict_emotion(spotify_id)
    '''
    song_name = "Baby Keem - family ties (with Kendrick Lamar)"
    path = f"Temp/audio/{song_name}.mp3"
    spotify_id = "7tFiyTwD0nx5a1eklYtX2J"
    if not os.path.exists(path):
        predict_emotion(spotify_id)

    spec = audio_to_logmel_array(path)
    preds, times = predict_spectrogram(spec, chunking=True)

    print("Chat GPT algorithm idea: ")
    chunkingData(preds, times)

    print("Arnav algorithm idea: ")
    chunkingData_stable_merge(preds, times)

    print("Claud algoirhtm idea: ")

    #0.6 ish for hotel cali
    chunkingData_dp(preds, times)
