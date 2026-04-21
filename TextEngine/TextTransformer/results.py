import torch
import numpy as np
import pandas as pd
from inference import load_model, predict
 
# created configuration variables 
CSV_PATH   = "goemotions_combined.csv"
MODEL_PATH = "emotion_transformer.pt"
VOCAB_PATH = "vocab.json"
THRESHOLD  = 0.5
TEST_SPLIT = 0.1
BATCH_SIZE = 64
OUTPUT     = "results.txt"  # set to None to skip saving

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

def load_test_data(csv_path):
    """
    reads the GoEmotions CSV and returns texts and their binary label matrix.
 
    arguments:
        csv_path (str): Path to the combined GoEmotions CSV.
 
    returns:
        texts (list[str]): List of input sentences.
        labels (np.ndarray): Binary array of shape (N, 28).
    """
    df = pd.read_csv(csv_path)
 
    if "text" not in df.columns:
        raise ValueError(f"CSV must have a 'text' column. Found: {list(df.columns)}")
 
    missing = [label for label in EMOTION_LABELS if label not in df.columns]
    if missing:
        print("[warn] Emotion columns not found by name. Falling back to last 28 columns as labels.")
        label_cols = df.columns[-28:]
    else:
        label_cols = EMOTION_LABELS
 
    texts  = df["text"].astype(str).tolist()
    labels = df[label_cols].values.astype(int)
    return texts, labels