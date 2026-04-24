import torch

from tokenizer import Tokenizer
from model import EmotionTransformer


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
        vocab_size=len(tokenizer.word_to_id),
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


def predict(texts, model, tokenizer, device, max_length=64, threshold=0.5):
    single = isinstance(texts, str)
    # if empty string, it would crash the tokenizer, so return empty results instead
    if not texts or (isinstance(texts, str) and texts.strip() == ""):
        return []

    if single:
        texts = [texts]

    batch_ids = [tokenizer.encode(text, max_length=max_length) for text in texts]
    # warn if any input exceeds max_length, which would be truncated by the tokenizer
    for i, text in enumerate(texts):
        raw_tokens = tokenizer.preprocess(text)
        if len(raw_tokens) > max_length:
            print(f"[warn] input {i} has {len(raw_tokens)} tokens, truncated to {max_length}")


    input_ids = torch.tensor(batch_ids, dtype=torch.long).to(device)
    attention_mask = (input_ids != 0).long()

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits)

    scores = probs.cpu().tolist()

    all_results = []
    for score_row in scores:
        results = [
            (EMOTION_LABELS[i], round(score_row[i], 4))
            for i in range(len(EMOTION_LABELS))
            if score_row[i] >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        # fallback: if nothing clears threshold, return highest confidence score
        if not results:
            top_idx = max(range(len(score_row)), key=lambda i: score_row[i])
            results = [(EMOTION_LABELS[top_idx], round(score_row[top_idx], 4))]
        
        all_results.append(results)


    return all_results[0] if single else all_results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(
        model_path="emotion_transformer.pt",
        vocab_path="vocab.json",
        device=device
    )

    text = "I feel scared and confused."
    results = predict(text, model, tokenizer, device, threshold=0.3)

    print("Input:", text)
    print("Predicted emotions:")
    for label, score in results:
        print(f"{label}: {score}")