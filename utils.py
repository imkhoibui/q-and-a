import os
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation except for periods and commas
    text = re.sub(r'[^\w\s,.]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespaces
    text = text.strip()
    
    return text

def load_glove():
    word_2_vec = {}

    print("Loading GloVe")

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove.6B.50d-relativized.txt")) as file:
        for line in file:
            l = line.split()
            word_2_vec[l[0]] = l[1:]

    print("Finish loading GloVe")
    return word_2_vec