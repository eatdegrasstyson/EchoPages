import pandas as pd
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from dataset import GoEmotionsDataset
from model import EmotionTransformer

csv_path = "goemotions_1.csv"
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