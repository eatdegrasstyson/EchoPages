from nrclex import NRCLex

text1 = "I feel excited but also a little nervous about tomorrow. I love learning new things."
text = 'I was so stressed last night because I was studying all day for my test in one hour. I was crying all day because of stress. After studying and taking the test though, I feel really good and happy with how I did. I feel like I a burden was lifted. Now I am super duper happy!'

'''
I was so stressed last night because I was studying all day for my test in one hour. 
I was crying all day because of stress. 
After studying and taking the test though, I feel really good and happy with how I did. 
I feel like I a burden was lifted. Now I am super duper happy!

Affect frequencies:
fear: 0.0
anger: 0.0
anticip: 0.0
trust: 0.1875
surprise: 0.0625
positive: 0.1875
negative: 0.125
sadness: 0.0625
disgust: 0.0
joy: 0.1875
anticipation: 0.1875

Raw emotion scores:
negative: 2
sadness: 1
anticipation: 3
joy: 3
positive: 3
surprise: 1
trust: 3

Emotion 1: ('trust', 0.1875)
Emotion 2: ('positive', 0.1875)
Emotion 3: ('joy', 0.1875)
Emotion 4: ('anticipation', 0.1875)
'''

# lex (NRCLex): analyzer object created from text
lex = NRCLex(text)

print("Affect frequencies:")
for emotion, score in lex.affect_frequencies.items():
    print(f"{emotion}: {score}")

print("\nRaw emotion scores:")
for emotion, score in lex.raw_emotion_scores.items():
    print(f"{emotion}: {score}")

print('\n')
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