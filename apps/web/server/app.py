import sys
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

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

    segments = []
    for sentence in sentences:
        emotions = predict_gems(model, sentence)
        dominant = max(emotions, key=emotions.get)
        segments.append({
            'text': sentence,
            'emotions': emotions,
            'dominant': dominant,
        })

    return jsonify({'segments': segments})


if __name__ == '__main__':
    app.run(port=5000, debug=False)
