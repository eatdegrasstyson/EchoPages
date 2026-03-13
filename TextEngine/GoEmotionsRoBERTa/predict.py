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
GOEMOTIONS_TO_GEMS = {
    "admiration":    "Wonder",
    "amusement":     "Joy",
    "anger":         "Tension",
    "annoyance":     "Tension",
    "approval":      None,
    "caring":        "Tenderness",
    "confusion":     None,
    "curiosity":     "Wonder",
    "desire":        "Tenderness",
    "disappointment": "Sadness",
    "disapproval":   "Tension",
    "disgust":       "Tension",
    "embarrassment": "Tension",
    "excitement":    "Power",
    "fear":          "Tension",
    "gratitude":     "Tenderness",
    "grief":         "Sadness",
    "joy":           "Joy",
    "love":          "Tenderness",
    "nervousness":   "Tension",
    "optimism":      "Joy",
    "pride":         "Power",
    "realization":   "Transcendence",
    "relief":        "Peacefulness",
    "remorse":       "Sadness",
    "sadness":       "Sadness",
    "surprise":      "Wonder",
    "neutral":       None,
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
    raise NotImplementedError


def predict_gems(model, text: str) -> dict:
    """End-to-end: text -> GEMS emotion vector."""
    raise NotImplementedError


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
