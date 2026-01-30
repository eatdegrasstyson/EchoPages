from nrclex import NRCLex

text = "I feel excited but also a little nervous about tomorrow. I love learning new things."

# lex (NRCLex): analyzer object created from text
lex = NRCLex(text)

print("Affect frequencies:", lex.affect_frequencies)
print("Raw emotion scores:", lex.raw_emotion_scores)
print("Top emotions:", lex.top_emotions)

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