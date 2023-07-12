import numpy as np, pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

import string
import re

#kaggle_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
kaggle_data = pd.read_csv('train.csv.zip')
kaggle_data = kaggle_data.rename(columns = {'comment_text':'tweet'})
kaggle_data['class'] = kaggle_data['toxic'] + kaggle_data['severe_toxic'] + kaggle_data['obscene'] +\
                            kaggle_data['threat'] + kaggle_data['insult'] + kaggle_data['identity_hate']
kaggle_data.loc[kaggle_data['class']>0,'class'] = 1
kaggle_data_1 = kaggle_data.copy()
del(kaggle_data)

kaggle_data_2 = pd.read_csv('https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv')
kaggle_data_2.loc[((kaggle_data_2['class']==0) | (kaggle_data_2['class']==1)),'class'] = 1
kaggle_data_2.loc[kaggle_data_2['class']==2,'class'] = 0


kaggle_data = pd.concat([kaggle_data_1[['tweet', 'class']],\
                            kaggle_data_2[['tweet', 'class']]])

kaggle_data['old_tweet'] = kaggle_data['tweet']

###########
trainDF = pd.concat([kaggle_data[kaggle_data['class']==0],
                     kaggle_data[kaggle_data['class']>0]])
texts = trainDF['tweet']#.head(10)
y = trainDF['class']#.head(10)

############
import nltk

nltk.download("wordnet")
#!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/ 

##############

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

#################

testDF_text = pd.read_csv('test.csv.zip')
print(testDF_text.columns)
testDF = pd.read_csv('test_labels.csv.zip')
print(testDF.columns)
testDF['class'] = testDF['toxic'] + testDF['severe_toxic'] + testDF['obscene'] +\
                            testDF['threat'] + testDF['insult'] + testDF['identity_hate']
testDF.loc[testDF['class']>0,'class'] = 1
testDF = testDF.loc[testDF['class']>=0]

testDF = pd.merge(testDF, testDF_text, on = 'id', how = 'left')

############
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(1000000)
print(sys.getrecursionlimit())

###########

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer
bow_vector = CountVectorizer(tokenizer = nltk_tokenizer, ngram_range=(1,2))
X = bow_vector.fit_transform(texts)

import pickle
filename = 'vectoriser_LSVC.pk'
with open('./'+filename, 'wb') as file:
    pickle.dump(bow_vector, file)

############
from sklearn.pipeline import Pipeline
model = LinearSVC(class_weight="balanced", dual=False, max_iter=100000, tol=1e-2)
model.fit(X,y)
cclf = CalibratedClassifierCV(base_estimator=model, cv='prefit', method='isotonic')
cclf.fit(X, y) 

##################

# Classification Report
from sklearn.metrics import classification_report

# Model Accuracy
print("LVC Model:\n")
y_test = testDF['class']

# Vectorize the text
X_test = bow_vector.transform(testDF['comment_text'])
predicted = cclf.predict(X_test)
print(classification_report(y_test, predicted, target_names = ['0','1']))

###################
# Save the Modle to file in the current working directory
filename = 'model_LSVC.pk'
with open('./'+filename, 'wb') as file:
    pickle.dump(cclf, file)