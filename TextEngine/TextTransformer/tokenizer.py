''' 
textEngine can't understand words
convert words into numbers
'''

import re
import json
class Tokenizer:

    def __init__(self):
        # maps a word to an integer(pad = padding and unk = unknown) 
        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        self.id_to_word = {
            0: "<PAD>",
            1: "<UNK>"
        }
    
    '''
    prepare text
    lowercase words and only keep words/numbers/apostrophes
    "I'm Happy!!!" -> ["i'm", "happy"]
    '''
    def preprocess(self, text):
        text = text.lower()
        tokens = re.findall(r"[a-z0-9']+", text)
        return tokens
    
    '''
    Build vocab from a list of texts
    if word not in vocab list, add it to list(word = next available integer id)
    '''
    def build_vocab(self, texts):
        for text in texts:
            tokens = self.preprocess(text)
            for token in tokens:
                if token not in self.word_to_id:
                    # this means its a new id, set it = next available integer id
                    new_id = len(self.word_to_id)
                    self.word_to_id[token] = new_id
                    self.id_to_word[new_id] = token

    '''
    Convert text into token Ids
    Unknown words become <UNK>
    Optionally pad/truncate to max_length 
        (transformers usually like when all the text block lengths are the same amount of words)
    returns list of tokenized words
    '''
    def encode(self, text, max_length = None):
        tokens = self.preprocess(text)
        tokens_ids = []

        for token in tokens:
            if token in self.word_to_id:
                tokens_ids.append(self.word_to_id[token])
            else:
                tokens_ids.append(self.word_to_id["UNK"])
        
        if max_length is not None:
            token_ids = self.pad_or_truncate(tokens_ids, max_length)
        
        return token_ids
    
    '''
    Convert token ids back into text
    returns string
    '''
    def decode(self, token_ids):
        words = []

        for token_id in token_ids:
            if token_id in self.id_to_word:
                words.append(self.id_to_word[token_id])
            else:
                words.append(self.id_to_word["UNK"])

        return " ".join(words)

    '''
    make every text block the same length(num of words)
    '''
    def pad_or_truncate(self, token_ids, max_length):
        
        if len(token_ids) < max_length:
            # padding needed (<Pad> tokens)
                padding_needed = max_length - len(token_ids)
                token_ids = token_ids + [self.word_to_id["<PAD>"]] * padding_needed
        else:
            #truncate
            token_ids = token_ids[:max_length]
        
        return token_ids
    
    '''
    save vocab to a JSON file
    '''
    def save_vocab(self, filepath):
        with open(filepath, "w") as file:
            json.dump(self.word_to_id, file, indent = 4)

    '''
    load vocan from a JSON file
    '''
    def load_vocab(self, filepath):
        with open(filepath, "r") as file:
            self.word_to_id = json.load(file)

        # id_to_word: rebuild reverse lookup after loading
        self.id_to_word = {}

        for word, token_id in self.word_to_id.items():
            self.id_to_word[int(token_id)] = word

        # Fix JSON issue: keys may load as strings in some cases
        self.word_to_id = {word: int(token_id) for word, token_id in self.word_to_id.items()}


# sample main to test
if __name__ == "__main__":
    sample_texts = [
        "I am happy today!",
        "This is so sad.",
        "I am excited and nervous."
    ]

    tokenizer = Tokenizer()
    tokenizer.build_vocab(sample_texts)

    print("Vocabulary:", tokenizer.word_to_id)
    print("Encoded:", tokenizer.encode("I am happy", max_length=6))
    print("Decoded:", tokenizer.decode(tokenizer.encode("I am happy", max_length=6)))

