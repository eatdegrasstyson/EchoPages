from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The old house stood silent, memories drifting like dust in the fading light."

gems_labels = [
    "wonder", "transcendence", "tenderness", "nostalgia",
    "peacefulness", "power", "joy", "tension", "sadness"
]

result = classifier(text, candidate_labels=gems_labels)

print("Top 3 GEMS categories with scores:")
for i in range(3):
    print(f"{result['labels'][i]}: {result['scores'][i]:.3f}")


# Terminal Test Code
# cd into your project folder or change the file path when running
# python3 could also be python for some users

# Install necessary libraries
# 1. pip install transformers
# 2. pip install torch
# 3. python3 and run test.py

