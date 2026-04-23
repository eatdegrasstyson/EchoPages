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