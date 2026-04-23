import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from dataset import GoEmotionsDataset
from model import EmotionTransformer

csv_paths = [
    "../GoEmotions/data/full_dataset/goemotions_1.csv",
    "../GoEmotions/data/full_dataset/goemotions_2.csv",
    "../GoEmotions/data/full_dataset/goemotions_3.csv"
]

max_length = 64
batch_size = 16
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df_list = [pd.read_csv(path) for path in csv_paths]
df = pd.concat(df_list, ignore_index=True)

combined_csv_path = "goemotions_combined.csv"
df.to_csv(combined_csv_path, index=False)

texts = df["text"].tolist()

tokenizer = Tokenizer()
tokenizer.build_vocab(texts)

dataset = GoEmotionsDataset(combined_csv_path, tokenizer, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Dataset size:", len(dataset))
print("Vocab size:", len(tokenizer.word_to_id))

model = EmotionTransformer(vocab_size=len(tokenizer.word_to_id)).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "emotion_transformer.pt")
tokenizer.save_vocab("vocab.json")

print("Model saved.")
print("Vocab saved.")