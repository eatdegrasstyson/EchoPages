import pandas as pd
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
    logits = model(input_ids)

    print("logits shape:", logits.shape)
    break