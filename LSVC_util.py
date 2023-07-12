import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

import string
import re

import nltk
nltk.download("wordnet")
nltk.download('stopwords')


# Create list of punctuation marks
punctuations = string.punctuation
def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct

# Remove URLs
def remove_urls(text):
    text = re.sub(r"\S*https?:\S*", "", text, flags=re.MULTILINE)
    text = re.sub(r"http?://\S+", "", text, flags=re.MULTILINE)
    return text

# remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(txt_tokenized):
    txt_clean = [word for word in txt_tokenized if word not in stopwords]
    return txt_clean

# porter stemming
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
def stemming(tokenized_text):
    text = [porter_stemmer.stem(word) for word in tokenized_text]
    return text

#
#nltk.download(wordnet)
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatization(token_txt):
    text = [wordnet_lemmatizer.lemmatize(word) for word in token_txt]
    return text

# Creat tokenizer function
def nltk_tokenizer(sentence):
    #print("sentence === ", sentence)
    
    sentence = sentence.lower()
    sentence = remove_urls(sentence)
    sentence = remove_punctuation(sentence)
    sentence = re.sub('[0-9]+', '', sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    tokens = nltk.word_tokenize(sentence)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    tokens = lemmatization(tokens)

    return tokens


