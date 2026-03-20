import sys
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

#Load the temp database:
import pandas as pd
import numpy as np
from numpy.linalg import norm

def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

# BASE_DIR = the backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths to CSVs
song_avg_path = os.path.normpath(os.path.join(BASE_DIR, '..', '..', '..', 'MusicEngine', 'DataSets', 'songDataBaseAveraged.csv'))
song_chunked_path = os.path.normpath(os.path.join(BASE_DIR, '..', '..', '..', 'MusicEngine', 'DataSets', 'songDataBaseChunked.csv'))

# Verify paths exist
if not os.path.exists(song_avg_path):
    raise FileNotFoundError(f"Averaged CSV not found at {song_avg_path}")
if not os.path.exists(song_chunked_path):
    raise FileNotFoundError(f"Chunked CSV not found at {song_chunked_path}")

# Load CSVs with fallback encoding
song_avg_df = pd.read_csv(song_avg_path, encoding='latin-1')
song_chunked_df = pd.read_csv(song_chunked_path, encoding='latin-1')

# GEMS-9 keys
GEMS_KEYS = ['Wonder','Transcendence','Tenderness','Nostalgia','Peacefulness','Power','Joy','Tension','Sadness']

# Function to find best song
def find_best_song(sentence_vector, use_chunked=False):
    df = song_chunked_df if use_chunked else song_avg_df
    sent_vec = np.array([sentence_vector[k] for k in GEMS_KEYS])

    # Cosine similarity
    song_vectors = df[GEMS_KEYS].to_numpy()
    cos_sim = song_vectors @ sent_vec / (norm(song_vectors, axis=1) * norm(sent_vec) + 1e-8)
    best_idx = np.argmax(cos_sim)
    best_song = df.iloc[best_idx]

    result = {
        'spotifyID': best_song['spotifyID'],
        'song_name': best_song['song_name']
    }
    if use_chunked:
        result['start'] = float(best_song['start'])
        result['end'] = float(best_song['end'])

    return result

#Dp chunking of sentences:
def chunk_text_dp(sentences, vectors,
                  lam=0.05,
                  target_words=25,
                  length_power=1.0,
                  ema_alpha=0.8,
                  min_words=5):
    """
    DP-based text segmentation.

    Minimizes:
        intra-chunk variance + length penalty

    Strongly penalizes small chunks to encourage paragraph-sized segments.
    """

    n = len(vectors)

    # --- Precompute word counts ---
    word_counts = np.zeros(n + 1)
    for i, s in enumerate(sentences):
        word_counts[i + 1] = word_counts[i] + len(s.split())

    # --- EMA smoothing ---
    smoothed = np.empty_like(vectors)
    smoothed[0] = vectors[0]
    for i in range(1, n):
        smoothed[i] = ema_alpha * vectors[i] + (1 - ema_alpha) * smoothed[i - 1]

    # --- Prefix sums for fast variance ---
    prefix_sum = np.zeros((n + 1, vectors.shape[1]))
    prefix_sq_sum = np.zeros(n + 1)

    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + smoothed[i]
        prefix_sq_sum[i + 1] = prefix_sq_sum[i] + np.dot(smoothed[i], smoothed[i])

    def segment_cost(i, j):
        length = j - i
        if length <= 1:
            return 0.0

        seg_sum = prefix_sum[j] - prefix_sum[i]
        seg_sq = prefix_sq_sum[j] - prefix_sq_sum[i]

        mean_sq_norm = np.dot(seg_sum, seg_sum) / (length * length)
        return seg_sq - length * mean_sq_norm

    # --- DP ---
    dp = np.full(n + 1, np.inf)
    dp[0] = 0.0
    parent = np.zeros(n + 1, dtype=int)

    for j in range(1, n + 1):
        for i in range(j):

            words_in_chunk = word_counts[j] - word_counts[i]

            # --- Hard constraint ---
            if words_in_chunk < min_words:
                continue

            # --- Length penalty (KEY PART) ---
            length_ratio = target_words / (words_in_chunk + 1e-8)
            length_penalty = lam * (length_ratio ** length_power)

            cost = dp[i] + segment_cost(i, j) + length_penalty

            if cost < dp[j]:
                dp[j] = cost
                parent[j] = i

    # --- Backtrack ---
    boundaries = []
    j = n
    while j > 0:
        i = parent[j]
        boundaries.append((i, j))
        j = i
    boundaries.reverse()

    return boundaries

def build_text_chunks(sentences, vectors, boundaries, GEMS_KEYS):
    chunks = []

    for start, end in boundaries:
        chunk_sentences = sentences[start:end]
        chunk_vecs = vectors[start:end]

        mean_vec = np.sum(chunk_vecs, axis=0)   # sum sentence vectors
        mean_vec = mean_vec / (np.sum(mean_vec) + 1e-8)   # normalize to sum = 1

        # mean_vec = np.mean(chunk_vecs, axis=0)
        # mean_vec = mean_vec / (norm(mean_vec) + 1e-8)s
        

        emotions = {
            GEMS_KEYS[i]: float(mean_vec[i])
            for i in range(len(GEMS_KEYS))
        }

        dominant = max(emotions, key=emotions.get)

        chunks.append({
            "text": " ".join(chunk_sentences),
            "start_idx": int(start),
            "end_idx": int(end),
            "word_count": int(sum(len(s.split()) for s in chunk_sentences)),
            "emotions": emotions,
            "dominant": dominant
        })

    return chunks


# Add TextEngine to path so we can import predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'TextEngine', 'GoEmotionsRoBERTa'))
from predict import load_model, predict_gems

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

print("Loading model...")
model = load_model()
print("Model loaded.")


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    #Sentence breakdown and emotion conversion
    segments = []
    sentence_vectors = []
    sentence_emotions = []

    for sentence in sentences:
        emotions = predict_gems(model, sentence)
        sentence_emotions.append(emotions)

        vec = np.array([emotions[k] for k in GEMS_KEYS])
        sentence_vectors.append(vec)

    sentence_vectors = np.array(sentence_vectors)

    #DP sentence chunking algorithm
    boundaries = chunk_text_dp(
        sentences,
        sentence_vectors,
        lam=0.2,
        target_words=25,
        length_power=1.5,
        ema_alpha=0.6,
        min_words=6
    )

    print("DP boundaries BEFORE fallback:", boundaries)
    if len(boundaries) == 0:
        boundaries = [(0, len(sentences))]
    print("DP boundaries AFTER fallback:", boundaries)

    # --- Step 3: build chunks ---
    chunks = build_text_chunks(
        sentences,
        sentence_vectors,
        boundaries,
        GEMS_KEYS
    )

    #Match best song to each chunk
    # for chunk in chunks:
    #     best_song = find_best_song(chunk['emotions'], use_chunked=False)
    #     chunk['matchedSong'] = best_song

    for chunk in chunks:
        # Match song using chunked database for finer time info
        best_song = find_best_song(chunk['emotions'], use_chunked=True)

        # Convert start/end times to minutes:seconds
        if 'start' in best_song and 'end' in best_song:
            best_song['start_formatted'] = format_time(best_song['start'])
            best_song['end_formatted'] = format_time(best_song['end'])

        chunk['matchedSong'] = best_song

    return jsonify({'segments': chunks})

    


if __name__ == '__main__':
    app.run(port=5000, debug=False)
