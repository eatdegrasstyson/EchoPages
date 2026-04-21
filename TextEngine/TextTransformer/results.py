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

def run_evaluation(model, tokenizer, texts, true_labels, device):
    """
    runs batched inference and converts scores to binary predictions.
 
    arguments s:
        model (EmotionTransformer): Loaded model in eval mode.
        tokenizer (Tokenizer): Tokenizer with vocab loaded.
        texts (list[str]): Input sentences.
        true_labels (np.ndarray): Ground truth binary labels, shape (N, 28).
        device (torch.device): Device to run inference on.
 
    returns:
        y_true (np.ndarray): Ground truth labels, shape (N, 28).
        y_pred (np.ndarray): Predicted binary labels, shape (N, 28).
    """
    all_preds = []
 
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_results = predict(batch_texts, model, tokenizer, device, threshold=0.0)
 
        for result_row in batch_results:
            score_dict = {label: score for label, score in result_row}
            binary_row = [
                1 if score_dict.get(label, 0.0) >= THRESHOLD else 0
                for label in EMOTION_LABELS
            ]
            all_preds.append(binary_row)
 
        print(f"  evaluated {min(i + BATCH_SIZE, len(texts))}/{len(texts)} samples...")
 
    return true_labels, np.array(all_preds)
 

def classification_report(y_true, y_pred, labels):
    """
    computes and formats per-class precision, recall, and F1 using numpy.
 
    arguments:
        y_true (np.ndarray): Ground truth binary labels, shape (N, num_classes).
        y_pred (np.ndarray): Predicted binary labels, shape (N, num_classes).
        labels (list[str]): Emotion label names.
 
    returns:
        str: Formatted report table.
    """
    lines = []
    lines.append(f"{'emotion':<20} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    lines.append("-" * 56)
 
    total_tp = total_fp = total_fn = 0
 
    for i, label in enumerate(labels):
        tp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        support = int(y_true[:, i].sum())
 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
 
        total_tp += tp
        total_fp += fp
        total_fn += fn
 
        lines.append(f"{label:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")
 
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1        = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
 
    lines.append("-" * 56)
    lines.append(f"{'micro avg':<20} {micro_precision:>10.4f} {micro_recall:>10.4f} {micro_f1:>10.4f} {int(y_true.sum()):>10}")
 
    return "\n".join(lines)

# main evaluation 
# sets up device, loads model and data, runs inference, and prints/saves the report.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
 
print("Loading model...")
model, tokenizer = load_model(MODEL_PATH, VOCAB_PATH, device)
 
print(f"Loading data from {CSV_PATH}...")
texts, labels = load_test_data(CSV_PATH)
 
split_idx = int(len(texts) * (1 - TEST_SPLIT))
texts  = texts[split_idx:]
labels = labels[split_idx:]
print(f"Evaluating on {len(texts)} samples (last {TEST_SPLIT*100:.0f}% of dataset).")
 
print("Running inference...")
y_true, y_pred = run_evaluation(model, tokenizer, texts, labels, device)
 
report = classification_report(y_true, y_pred, EMOTION_LABELS)
 
print("\n===== Classification Report =====")
print(report)
 
if OUTPUT:
    with open(OUTPUT, "w") as f:
        f.write(report)
    print(f"Saved to {OUTPUT}")