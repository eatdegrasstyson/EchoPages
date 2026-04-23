import torch
import pandas as pd
from inference import load_model, predict
 
'''
This script performs batch inference on a set of input sentences and saves the results to a CSV file
The input can be a .txt file (one sentence per line) or a .csv file (must have a 'text' column)
The output CSV will have columns: 'text', 'emotion', 'score'
'''

# cofniguration variables
MODEL_PATH = "emotion_transformer.pt"
VOCAB_PATH = "vocab.json"
THRESHOLD  = 0.3
BATCH_SIZE = 64
INPUT_FILE = "input.txt"   # .txt (one sentence per line) or .csv (must have a 'text' column)
OUTPUT_FILE = "batch_results.csv"
 
 
def load_input(filepath):
    """
    loads sentences from a .txt or .csv file.
 
    .txt: expects one sentence per line.
    .csv: expects a 'text' column.
 
    arguements: filepath (str): Path to input file.
 
    returns: list[str]: List of sentences.
    """
    if filepath.endswith(".txt"):
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line]  # drop empty lines
 
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        if "text" not in df.columns:
            raise ValueError(f"CSV must have a 'text' column. Found: {list(df.columns)}")
        return df["text"].astype(str).tolist()
 
    else:
        raise ValueError(f"Unsupported file type: {filepath}. Use .txt or .csv")
    
def run_batch(texts, model, tokenizer, device):
    """
    Runs predict() over all texts in batches.
 
    Args:
        texts (list[str]): Input sentences.
        model (EmotionTransformer): Loaded model in eval mode.
        tokenizer (Tokenizer): Tokenizer with vocab loaded.
        device (torch.device): Device to run inference on.
 
    Returns:
        list[list[tuple]]: One list of (emotion, score) tuples per input text.
    """
    all_results = []
 
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        results = predict(batch, model, tokenizer, device, threshold=THRESHOLD)
        all_results.extend(results)
        print(f"  processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} sentences...")
 
    return all_results
 
 
def save_results(texts, all_results, output_path):
    """
    Saves predictions to a CSV with one row per input sentence.
 
    Columns: text, emotions, top_emotion, top_score
 
    Args:
        texts (list[str]): Original input sentences.
        all_results (list): Predictions from run_batch().
        output_path (str): Path to save the output CSV.
    """
    rows = []
    for text, results in zip(texts, all_results):
        if results:
            top_emotion, top_score = results[0]
            emotions_str = ", ".join(f"{label}({score})" for label, score in results)
        else:
            top_emotion, top_score = "none", 0.0
            emotions_str = "none"
 
        rows.append({
            "text":        text,
            "emotions":    emotions_str,
            "top_emotion": top_emotion,
            "top_score":   top_score
        })
 
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(rows)} results to {output_path}")

# main execution block
# loads model and tokenizer, runs batch inference, and saves results to CSV
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Loading model...")
    model, tokenizer = load_model(MODEL_PATH, VOCAB_PATH, device)
    
    print(f"Loading input from {INPUT_FILE}...")
    texts = load_input(INPUT_FILE)
    print(f"Loaded {len(texts)} sentences.")
    
    print("Running batch inference...")
    all_results = run_batch(texts, model, tokenizer, device)
    
    save_results(texts, all_results, OUTPUT_FILE)