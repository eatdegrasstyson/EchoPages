from transformers import pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMS_LABELS = [
    "Wonder",
    "Transcendence",
    "Tenderness",
    "Nostalgia",
    "Peacefulness",
    "Power",
    "Joy",
    "Tension",
    "Sadness",
]

# GoEmotions 27 labels (+ neutral) -> GEMS bucket mapping.
# Each GoEmotions label maps to one GEMS category (or None to discard).
# Weights and exact assignments will be tuned in a later step.
GOEMOTIONS_TO_GEMS = { #W,Tr, Ten, N, Pea, Pow, J, Tension, S

    "admiration":    [1,0,0,0,0,0,0,0,0],
    "amusement":     [0,0,0,0,0,0,1,0,0],
    "anger":         [0,0,0,0,0,0,0,1,0],
    "annoyance":     [0,0,0,0,0,0,0,1,0],
    "approval":      [0,0,0,0,0,0,0,0,0],
    "caring":        [0,0,1,0,0,0,0,0,0],
    "confusion":     [0,0,0,0,0,0,0,0,0],
    "curiosity":     [1,0,0,0,0,0,0,0,0],
    "desire":        [0,0,1,0,0,0,0,0,0],
    "disappointment":[0,0,0,0,0,0,0,0,1],
    "disapproval":   [0,0,0,0,0,0,0,1,0],
    "disgust":       [0,0,0,0,0,0,0,1,0],
    "embarrassment": [0,0,0,0,0,0,0,1,0],
    "excitement":    [0,0,0,0,0,1,0,0,0],
    "fear":          [0,0,0,0,0,0,0,1,0],
    "gratitude":     [0,0,1,0,0,0,0,0,0],
    "grief":         [0,0,0,0,0,0,0,0,1],
    "joy":           [0,0,0,0,0,0,1,0,0],
    "love":          [0,0,1,0,0,0,0,0,0],
    "nervousness":   [0,0,0,0,0,0,0,1,0],
    "optimism":      [0,0,0,0,0,0,1,0,0],
    "pride":         [0,0,0,0,0,1,0,0,0],
    "realization":   [0,1,0,0,0,0,0,0,0],
    "relief":        [0,0,0,0,1,0,0,0,0],
    "remorse":       [0,0,0,0,0,0,0,0,1],
    "sadness":       [0,0,0,0,0,0,0,0,1],
    "surprise":      [1,0,0,0,0,0,0,0,0],
    "neutral":       [0,0,0,0,0,0,0,0,0],
}

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model():
    """Load the SamLowe/roberta-base-go_emotions pipeline."""
    return pipeline(
        task="text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,   # return scores for all 28 labels, not just the top one
    )


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_goemotions(model, text: str) -> dict:
    """
    Run the model on `text` and return a dict of
    {goemotions_label: probability} for all 28 labels.
    """
    results = model(text)[0]  # list of {"label": str, "score": float}
    return {entry["label"]: entry["score"] for entry in results}


def aggregate_to_gems(goemotions_scores: dict) -> dict:
    """
    Aggregate GoEmotions label probabilities into GEMS buckets
    using GOEMOTIONS_TO_GEMS. Returns a dict of {gems_label: score}
    normalized so all 9 values sum to 1.
    """
    gems_scores = [0.0] * len(GEMS_LABELS)

    for label, prob in goemotions_scores.items():
        vec = GOEMOTIONS_TO_GEMS.get(label)
        if vec is None:
            continue
        for i in range(len(GEMS_LABELS)):
            gems_scores[i] += prob * vec[i]

    total = sum(gems_scores)
    if total > 0:
        gems_scores = [v / total for v in gems_scores]

    return {label: gems_scores[i] for i, label in enumerate(GEMS_LABELS)}


def predict_gems(model, text: str) -> dict:
    """End-to-end: text -> GEMS emotion vector."""
    go_scores = predict_goemotions(model, text)
    return aggregate_to_gems(go_scores)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "I call it independence, but sometimes it's just fear dressed up nicely."

    print("Loading model...")
    model = load_model()

    print(f"\nText:\n\t{sample}\n")

    print("GoEmotions scores:")
    go_scores = predict_goemotions(model, sample)
    for label, score in sorted(go_scores.items(), key=lambda x: -x[1]):
        print(f"\t{label}: {score:.4f}")

    print("\nGEMS vector:")
    gems = predict_gems(model, sample)
    for label, score in gems.items():
        print(f"\t{label}: {score:.4f}")