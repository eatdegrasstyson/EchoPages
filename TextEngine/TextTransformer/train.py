import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from dataset import GoEmotionsDataset
from model import EmotionTransformer

csv_path = "../GoEmotions/data/full_dataset/goemotions_1.csv"
max_length = 64
batch_size = 16

# load text so tokenizer can build vocab
df = pd.read_csv(csv_path)
texts = df["text"].tolist()

tokenizer = Tokenizer()
tokenizer.build_vocab(texts)

dataset = GoEmotionsDataset(csv_path, tokenizer, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Dataset size:", len(dataset))
print("Vocab size:", len(tokenizer.word_to_id))

for batch in dataloader:
    print("input_ids shape:", batch["input_ids"].shape)
    print("labels shape:", batch["labels"].shape)
    print(batch["input_ids"][0])
    print(batch["labels"][0])
    break

model = EmotionTransformer(vocab_size=len(tokenizer.word_to_id))

for batch in dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].to(device)
    logits = model(input_ids, attention_mask=attention_mask)

    print("logits shape:", logits.shape)
    break


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = EmotionTransformer(vocab_size=len(tokenizer.word_to_id)).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1

model.train()

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
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