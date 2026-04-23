import pandas as pd
import numpy as np

#GEMS-9 emotion order MUST match the model output columns
EMOTIONS = [
    "Wonder", "Transcendence", "Tenderness", "Nostalgia",
    "Peacefulness", "Power", "Joy", "Tension", "Sadness"
]

def load_predictions(pred_csv):
    df = pd.read_csv(pred_csv)
    
    #Ensure correct column order
    vectors = df[EMOTIONS].values
    ids = df["spotifyID"].values
    
    return ids, vectors

def load_labels(label_csv):
    df = pd.read_csv(label_csv)
    
    id_to_label = {}
    for _, row in df.iterrows():
        id_to_label[row["spotify_id"]] = row["gems9_emotion"]
    
    return id_to_label

def one_hot(emotion):
    vec = np.zeros(len(EMOTIONS))
    idx = EMOTIONS.index(emotion)
    vec[idx] = 1.0
    return vec

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evaluate(pred_csv, label_csv):
    ids, preds = load_predictions(pred_csv)
    labels = load_labels(label_csv)
    
    total = 0
    top1_correct = 0
    top3_correct = 0
    cos_sims = []
    
    for i in range(len(ids)):
        sid = ids[i]
        
        if sid not in labels:
            continue
        
        pred_vec = preds[i]
        true_emotion = labels[sid]
        true_vec = one_hot(true_emotion)
        
        # Cosine similarity
        cos = cosine_similarity(pred_vec, true_vec)
        cos_sims.append(cos)
        
        # Top-k predictions
        top_indices = np.argsort(pred_vec)[::-1]
        top1 = EMOTIONS[top_indices[0]]
        top3 = [EMOTIONS[idx] for idx in top_indices[:3]]
        
        # Accuracy checks
        if top1 == true_emotion:
            top1_correct += 1
        
        if true_emotion in top3:
            top3_correct += 1
        
        total += 1
    
    print("Total samples:", total)
    print("Average Cosine Similarity:", np.mean(cos_sims))
    print("Top-1 Accuracy:", top1_correct / total)
    print("Top-3 Accuracy:", top3_correct / total)


# ==== USAGE ====
evaluate("DataSets/songDataBaseAveragedDemo.csv", "DataSets/CustomDataset.csv")