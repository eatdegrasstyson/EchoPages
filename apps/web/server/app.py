import sys
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

#Load the temp database:
import pandas as pd
import numpy as np
from numpy.linalg import norm

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


# Add TextEngine to path so we can import predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'TextEngine', 'GoEmotionsRoBERTa'))
from predict import load_model, predict_gems

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

print("Loading model...")
model = load_model()
print("Model loaded.")


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
    }), 200


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    MAX_CHARS = 20_000
    if len(text) > MAX_CHARS:
        return jsonify({'error': f'Text too long (max {MAX_CHARS} characters)'}), 400

    paragraphs = re.split(r'\n{2,}', text)
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if para:
            sentences.extend(s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip())

    segments = []
    for sentence in sentences:
        emotions = predict_gems(model, sentence)
        dominant = max(emotions, key=emotions.get)
        best_song = find_best_song(emotions, use_chunked=False)
        segments.append({
            'text': sentence,
            'emotions': emotions,
            'dominant': dominant,
            'matchedSong': best_song,
        })

    return jsonify({'segments': segments})

    


if __name__ == '__main__':
    app.run(port=5000, debug=False)
