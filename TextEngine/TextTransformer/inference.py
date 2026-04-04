import torch
import torch.nn as nn

from tokenizer import Tokenizer
from model import EmotionTransformer


# emotion labels for reference
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

def load_model(model_path, vocab_path, device):
    tokenizer = Tokenizer()
    tokenizer.load_vocab(vocab_path)

    model = EmotionTransformer(
        vocab_size=tokenizer.vocab_size(),
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_classes=len(EMOTION_LABELS),
        max_length=64,
        dropout=0.1
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer


"""
texts: a single string or a list of strings
returns: a single result list if given a string, list of result lists if given a list
"""
def predict(texts, model, tokenizer, device, max_length=64, threshold=0.0):
    single = isinstance(texts, str)
    if single:
        texts = [texts]

    batch_ids = [tokenizer.encode(t, max_length=max_length) for t in texts]
    input_ids = torch.tensor(batch_ids, dtype=torch.long).to(device)
    attention_mask = (input_ids != 0).long()

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)

    scores = logits.cpu().tolist()

    all_results = []
    for score_row in scores:
        results = [
            (EMOTION_LABELS[i], round(score_row[i], 4))
            for i in range(len(EMOTION_LABELS))
            if score_row[i] >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        all_results.append(results)

    return all_results[0] if single else all_results

if __name__ == "__main__":
    # example usage
    from dataset import GoEmotionsDataset  

    # just uses the first 100 rows of the dataset for testing of tokenizer and model loading with no train.py yet and no trained model, so results will be random
    # make sure to have a "goemotions_sample.csv" file with the same format as the original dataset for this to work
    tokenizer = Tokenizer()
    tokenizer.build_vocab(["I'm happy", "this is sad", "so excited"])
    tokenizer.save_vocab("vocab.json")
    # hardcoded test sentences and vocab for testing, replace with actual model and vocab paths when completed