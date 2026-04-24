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

combined_csv_path = "goemotions_combined.csv"
max_length = 64
batch_size = 16
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# loads and combines CSV files, returns a DataFrame
def load_data(csv_paths, combined_csv_path):
    df_list = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(combined_csv_path, index=False)
    return df

# training loop function to be called in main block, separated for clarity and potential reuse
def train(model, dataloader, loss_fn, optimizer, device, num_epochs):
    log = []
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss   = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        log.append(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    with open("training_log.txt", "w") as f:
        f.write("\n".join(log))
    print("Training log saved.")

# main execution block, which loads data, sets up model and training components, and runs the training loop
def main():
    df = load_data(csv_paths, combined_csv_path)
    texts = df["text"].tolist()

    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts)

    dataset    = GoEmotionsDataset(combined_csv_path, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Dataset size:", len(dataset))
    print("Vocab size:",   len(tokenizer.word_to_id))

    model     = EmotionTransformer(vocab_size=len(tokenizer.word_to_id)).to(device)
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, loss_fn, optimizer, device, num_epochs)

    torch.save(model.state_dict(), "emotion_transformer.pt")
    tokenizer.save_vocab("vocab.json")

    print("Model saved.")
    print("Vocab saved.")

if __name__ == "__main__":
    main()