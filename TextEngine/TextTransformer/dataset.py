import pandas as pd
import torch
from torch.utils.data import Dataset


class GoEmotionsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length = 64):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # these are the actual label columns in CSV
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse",
            "sadness", "surprise", "neutral"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        text = row["text"]

        input_ids = self.tokenizer.encode(text, max_length=self.max_length)

        label_values = row[self.emotion_labels].values.astype("float32")

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(label_values, dtype=torch.float32)

        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "labels" : labels
        }

        