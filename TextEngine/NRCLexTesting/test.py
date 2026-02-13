from nrclex import NRCLex

text = "This was a complete waste of money. It broke after one use and customer service was terrible."
# lex (NRCLex): analyzer object created from text
lex = NRCLex(text)

print("Text:\n\t" + text + "\n")
print("Affect frequencies:")
for emotion, score in lex.affect_frequencies.items():
    print(f"\t{emotion}: {score}")

print("\nRaw emotion scores:")
for emotion, score in lex.raw_emotion_scores.items():
    print(f"\t{emotion}: {score}")

print("\n")
count = 0
for emotion in lex.top_emotions:
    count += 1
    print(f"Emotion {count}: {emotion}")

# Terminal Test Code

# cd into the NRCLexTesting folder or change the file path when running 
# python3 could also be python for some users
# - Test to see if you can run
# 1. python3 test.py
# - will probably have to run this command first to download the neccessary NRCLex data
# - Step 2 might be optional for non mac users
# 2. open "/Applications/Python 3.14/Install Certificates.command"
# 3. python3 -m textblob.download_corpora
# 4. python3 test.py