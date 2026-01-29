from nrclex import NRCLex

text = "I feel excited but also a little nervous about tomorrow. I love learning new things."

# lex (NRCLex): analyzer object created from text
lex = NRCLex(text)

print("Affect frequencies:", lex.affect_frequencies)
print("Raw emotion scores:", lex.raw_emotion_scores)
print("Top emotions:", lex.top_emotions)

