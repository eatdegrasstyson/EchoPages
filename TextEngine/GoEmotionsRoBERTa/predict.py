from transformers import pipeline
import numpy as np

# ---------------------------------------------------------------------------
# GEMS-9 anchor sentences
# 6 sentences per category — varied phrasing to reduce variance
# Used to determine how to convert sentence predictions to gems. (more accurate then 0,1 rigid matrix)
# ---------------------------------------------------------------------------

GEMS_ANCHORS = {
    "Wonder": [
        "I gazed up at the night sky, breathless at the infinite stars above me.",
        "The sheer scale of the canyon left me speechless and small.",
        "I stumbled upon a hidden waterfall deep in the forest and could not believe my eyes.",
        "The complexity of a single cell under the microscope filled me with awe.",
        "Standing at the edge of the ocean I felt the mystery of everything unknown.",
        "The northern lights danced overhead and I forgot how to speak.",
    ],
    "Transcendence": [
        "In that moment I felt deeply connected to something far greater than myself.",
        "Time dissolved and I felt a profound unity with the universe around me.",
        "A wave of pure clarity washed over me and everything made perfect sense.",
        "I felt my sense of self dissolve into something boundless and eternal.",
        "It was as if the boundary between me and the world simply disappeared.",
        "I experienced a stillness so deep it felt like touching the infinite.",
    ],
    "Tenderness": [
        "She held the sleeping child close and felt a love beyond words.",
        "He gently wiped the tears from her cheek without saying anything.",
        "The old dog rested its head on her lap and she stroked it softly.",
        "They sat together in comfortable silence, just glad to be near each other.",
        "She left a small note in his lunchbox that just said I love you.",
        "He took her hand carefully as if holding something precious.",
    ],
    "Nostalgia": [
        "The smell of that old kitchen took me straight back to my grandmother's house.",
        "I found my childhood toys in a box and ached for those simple days.",
        "That song came on and suddenly I was seventeen again.",
        "Looking at the old photographs I felt the weight of everything that had passed.",
        "I drove past my old school and was overwhelmed by how much time had gone.",
        "I longed for the slow summer afternoons that would never come back.",
    ],
    "Peacefulness": [
        "I sat by the still lake at dawn and felt completely at ease.",
        "The quiet of the early morning settled over everything like a soft blanket.",
        "There was nowhere to be and nothing to do and it felt perfect.",
        "I closed my eyes on the warm grass and let all my worries drift away.",
        "The gentle sound of rain on the roof filled me with calm.",
        "Everything felt unhurried and soft and exactly as it should be.",
    ],
    "Power": [
        "She stood at the podium and commanded the entire room without raising her voice.",
        "He crossed the finish line and felt invincible.",
        "They rose up together and refused to be pushed aside any longer.",
        "She realized no one could take away what she had built.",
        "The team stormed back from nothing and won in the final seconds.",
        "He faced his fear head on and felt something fierce unlock inside him.",
    ],
    "Joy": [
        "She burst out laughing and could not stop no matter how hard she tried.",
        "Everything felt light and bright and impossibly good.",
        "They danced in the kitchen at midnight for absolutely no reason.",
        "He got the news and jumped around the room like a child.",
        "The whole day felt like a gift and she soaked up every moment.",
        "Pure simple happiness filled the room and everyone felt it.",
    ],
    "Tension": [
        "She heard a sound downstairs and froze completely still.",
        "His hands were shaking as he waited for the call that would decide everything.",
        "The silence between them was unbearable and loaded with things unsaid.",
        "Something was very wrong but no one could say exactly what it was.",
        "Every footstep in the hallway made her heart pound harder.",
        "He sat in the waiting room unable to breathe properly.",
    ],
    "Sadness": [
        "She sat alone in the empty house and cried without knowing when she would stop.",
        "The grief came in waves and sometimes he could not get out of bed.",
        "She stared at the chair he used to sit in and felt the absence like a wound.",
        "Nothing felt worth doing anymore and the days blurred into each other.",
        "He read the old messages and felt the loss all over again.",
        "The world kept moving but she felt completely still and left behind.",
    ],
}


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

CustomEmbeddingMapping={  #Wonder, Trance, Tender, Nosta,  Peace,  Power,  Joyful, Tension, Sadness
        "admiration":     [0.1886, 0.1023, 0.1588, 0.0156, 0.1311, 0.2071, 0.1889, 0.0029, 0.0046],
        "amusement":      [0.0204, 0.0191, 0.0138, 0.0430, 0.0164, 0.0344, 0.8319, 0.0098, 0.0112],
        "anger":          [0.0809, 0.0634, 0.0859, 0.0836, 0.0355, 0.1168, 0.0656, 0.2675, 0.2008],
        "annoyance":      [0.0699, 0.0496, 0.0262, 0.0774, 0.0242, 0.0629, 0.0308, 0.3770, 0.2819],
        "approval":       [0.0132, 0.3042, 0.0695, 0.0428, 0.3599, 0.1298, 0.0489, 0.0173, 0.0144],
        "caring":         [0.0080, 0.0600, 0.3905, 0.0150, 0.3714, 0.0430, 0.0552, 0.0290, 0.0279],
        "confusion":      [0.4817, 0.0469, 0.0108, 0.0278, 0.0170, 0.0169, 0.0116, 0.3578, 0.0295],
        "curiosity":      [0.7727, 0.0359, 0.0171, 0.0368, 0.0144, 0.0128, 0.0220, 0.0544, 0.0337],
        "desire":         [0.0177, 0.0234, 0.0666, 0.7697, 0.0251, 0.0296, 0.0244, 0.0260, 0.0176],
        "disappointment": [0.0880, 0.0283, 0.0082, 0.1709, 0.0084, 0.0159, 0.0056, 0.2804, 0.3943],
        "disapproval":    [0.0532, 0.0274, 0.0434, 0.0501, 0.0563, 0.1806, 0.0315, 0.4215, 0.1360],
        "disgust":        [0.0864, 0.0539, 0.0483, 0.1301, 0.0312, 0.0639, 0.0312, 0.4135, 0.1414],
        "embarrassment":  [0.3043, 0.1004, 0.0206, 0.1158, 0.0327, 0.0798, 0.0317, 0.2301, 0.0845],
        "excitement":     [0.2913, 0.1584, 0.0171, 0.0868, 0.0374, 0.1147, 0.2805, 0.0094, 0.0044],
        "fear":           [0.0540, 0.0648, 0.0169, 0.0519, 0.0271, 0.5184, 0.0117, 0.1879, 0.0673],
        "gratitude":      [0.0392, 0.2313, 0.2022, 0.0292, 0.1561, 0.0980, 0.2157, 0.0117, 0.0166],
        "grief":          [0.0493, 0.0827, 0.0501, 0.1489, 0.0597, 0.0386, 0.0236, 0.1912, 0.3559],
        "joy":            [0.0185, 0.1393, 0.1426, 0.0156, 0.2832, 0.0301, 0.3642, 0.0034, 0.0029],
        "love":           [0.0119, 0.0193, 0.8812, 0.0129, 0.0249, 0.0046, 0.0313, 0.0071, 0.0067],
        "nervousness":    [0.0920, 0.1381, 0.0232, 0.0968, 0.1195, 0.0491, 0.0200, 0.3595, 0.1019],
        "optimism":       [0.0634, 0.1299, 0.1272, 0.1495, 0.1339, 0.2368, 0.0732, 0.0471, 0.0390],
        "pride":          [0.0346, 0.3371, 0.0436, 0.0371, 0.1256, 0.3048, 0.1009, 0.0092, 0.0071],
        "realization":    [0.1064, 0.2894, 0.0116, 0.2589, 0.0415, 0.1937, 0.0112, 0.0444, 0.0427],
        "relief":         [0.0132, 0.3067, 0.0872, 0.0303, 0.3859, 0.0623, 0.0868, 0.0150, 0.0127],
        "remorse":        [0.0891, 0.0635, 0.1129, 0.1282, 0.0869, 0.1044, 0.0309, 0.1690, 0.2152],
        "sadness":        [0.0104, 0.0088, 0.0183, 0.2214, 0.0154, 0.0062, 0.0031, 0.2488, 0.4676],
        "surprise":       [0.6562, 0.0677, 0.0016, 0.2395, 0.0022, 0.0122, 0.0085, 0.0068, 0.0053],
        "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

#Upgrades goemtions to gems matrix 
def build_gems_mapping(model) -> dict:
    """
    Runs each anchor sentence through the GoEmotions classifier and averages
    the output vectors per GEMS category to produce a soft mapping matrix.
    
    Returns: {gems_label: {goemotions_label: averaged_weight}}
    """
    #print("Building GEMS mapping from anchors...")
    mapping = {}

    for gems_label, sentences in GEMS_ANCHORS.items():
        all_scores = []
        for sentence in sentences:
            go_scores = predict_goemotions(model, sentence)
            all_scores.append(go_scores)

        # Average GoEmotions distribution across all anchor sentences
        averaged = {}
        for go_label in all_scores[0]:
            averaged[go_label] = sum(s[go_label] for s in all_scores) / len(all_scores)

        mapping[gems_label] = averaged
        
        # Print so you can inspect the dominant GoEmotions labels per GEMS category
        top = sorted(averaged.items(), key=lambda x: -x[1])[:5]
        print(f"  {gems_label}: {', '.join(f'{k}({v:.3f})' for k,v in top)}")

    return mapping

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

#Aggregates to gems using ay learned mapping input
def aggregate_to_gems_learned(goemotions_scores: dict, mapping: dict) -> dict:
    gems_scores = [0.0] * len(GEMS_LABELS)

    for go_label, prob in goemotions_scores.items():
        weights = mapping.get(go_label)
        if weights is None:
            continue
        for i in range(len(GEMS_LABELS)):
            gems_scores[i] += prob * weights[i]

    total = sum(gems_scores)
    if total > 0:
        gems_scores = [v / total for v in gems_scores]

    return {label: gems_scores[i] for i, label in enumerate(GEMS_LABELS)}

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

#Function overload with specified mapping
def predict_gems(model, text: str, mapping: dict = CustomEmbeddingMapping) -> dict:
    """End-to-end: text -> GEMS-9 vector using learned mapping."""
    go_scores = predict_goemotions(model, text)
    return aggregate_to_gems_learned(go_scores, mapping)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading model...")
    model = load_model()

    GEMS_LABELS_LOCAL = ['Wonder','Transcendence','Tenderness','Nostalgia','Peacefulness','Power','Joy','Tension','Sadness']
    GO_LABELS = ['admiration','amusement','anger','annoyance','approval','caring','confusion',
                 'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
                 'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
                 'pride','realization','relief','remorse','sadness','surprise','neutral']

    mapping = build_gems_mapping(model)

    # Transpose: {go_label -> [w0..w8]} and normalize each row to sum to 1
    print("\nCustomEmbeddingMapping = {  #W, Tr, Ten, N, Pea, Pow, J, Tension, S")
    for go_label in GO_LABELS:
        row = [mapping[gems_label].get(go_label, 0.0) for gems_label in GEMS_LABELS_LOCAL]
        total = sum(row)
        row = [v / total for v in row] if total > 0 else row
        formatted = ', '.join(f'{v:.4f}' for v in row)
        print(f'    "{go_label}": [{formatted}],')
    print("}")

    # --- Test sentences, one per GEMS category ---
    TEST_SENTENCES = {
        "Wonder":        "The vast ocean stretched endlessly before me and I felt tiny and amazed.",
        "Transcendence": "I felt my ego dissolve into something larger than myself.",
        "Tenderness":    "She brushed the hair from his face and smiled.",
        "Nostalgia":     "The old song came on and I remembered my childhood bedroom.",
        "Peacefulness":  "I sat on the porch in the early morning and felt relaxed.",
        "Power":         "She stepped up to the microphone and owned every inch of the room.",
        "Joy":           "He ran out of the building laughing, unable to contain his happiness.",
        "Tension":       "The door creaked open slowly and every muscle in her body went rigid.",
        "Sadness":       "He sat by the grave long after everyone else had gone home.",
    }

    # Build transposed mapping for predict_gems_learned
    transposed = {}
    for go_label in GO_LABELS:
        row = [mapping[gems_label].get(go_label, 0.0) for gems_label in GEMS_LABELS_LOCAL]
        transposed[go_label] = row

    print("\n--- Test sentence GEMS vectors ---")
    for expected_gem, sentence in TEST_SENTENCES.items():
        gems_vec = predict_gems(model, sentence, transposed)
        dominant = max(gems_vec, key=gems_vec.get)
        correct = "✓" if dominant == expected_gem else "✗"
        print(f"\n[{correct}] Expected: {expected_gem} | Got: {dominant}")
        print(f"    \"{sentence}\"")
        for label, score in gems_vec.items():
            bar = "█" * int(score * 40)
            print(f"    {label:15s} {score:.3f} {bar}")