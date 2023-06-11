import sys
import re
import pandas as pd

import nltk
nltk.download(['punkt','wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
    
def tokenize(text):
    '''
    Process the text data by normalizing it, and removing punctuation
    '''
    #Normalize the text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stopwords.words("english")]

    return lemmed
