import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

#### data prep ####
import string
import re 
import spacy
import contextualSpellCheck

global token_couner
token_couner = 0

# Load the English library from SpaCy
nlp = spacy.load("en_core_web_sm")

# Add contextual spell check to pipeline
nlp.add_pipe("contextual spellchecker", config={"max_edit_dist": 5})    
#contextualSpellCheck.add_to_pipe(nlp)

# Create list of punctuation marks
punctuations = string.punctuation

# Create list of stopwords from spaCy
stopwords = spacy.lang.en.stop_words.STOP_WORDS
#####
# end data prep
#####

###########
# Remove URLs
def remove_urls(text):
    text = re.sub(r"\S*https?:\S*", "", text, flags=re.MULTILINE)
    text = re.sub(r"http?://\S+", "", text, flags=re.MULTILINE)
    return text

# Creat tokenizer function
def spacy_tokenizer(sentence):
    print("sentence === ", sentence)
    
    sentence = remove_urls(sentence)
    
    # remove regex
    sentence = re.sub(r'[^\w\s]', '', sentence)
    #print("remove rgex step === ",sentence)
    
    
    # Create token object from spacy
    docs = nlp(sentence)

    
    # Correct spelling
    #tokens = docs._.outcome_spellCheck
    #tokens = nlp(tokens)
    #print("tokens ====== ",tokens)

    # Lemmatize each token and convert each token into lowercase
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "PROPN" else word.lower_ for word in docs]
    #print("tokens ====== ",tokens)
    
    # Remove links
    tokens = [remove_urls(word) for word in tokens]
    #print("tokens ====== ",tokens)
    
    # Remove stopwords
    print("tokens ====== ",tokens)
    tokens = [word for word in tokens if word not in stopwords]
    tokens = [word for word in tokens if word not in punctuations]
    tokens = [word for word in tokens if word not in list(string.ascii_lowercase)]
    
    print("tokens ====== ",tokens)
    
    global token_couner
    token_couner = token_couner + 1
    print(token_couner)

    # return preprocessed list of tokens
    return tokens
#######

#model = pickle.load(open('model.pkl','rb'))
# Open saved model, and directly make the prediction with new data
filename = 'Pickle_NB_Model.pk'
with open('./'+filename ,'rb') as f:
    model = pickle.load(f)
#loaded_model.predict(X_test.loc[0:15])


@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/api',methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     prediction = model.predict([[np.array(data['exp'])]])
#     output = prediction[0]
#     return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():

    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    tweet = request.form.values()
    print("tweet---- ", tweet)
    prediction = model.predict(tweet)

    #output = round(prediction[0], 2)
    output = prediction
    print("tyep(output)====> ", output)

    if output==1:
        print("Can't accept the tweet {}".format(output))
        return render_template('index.html', prediction_text="Can't accept the tweet {}".format(output))
    else:
        print("Can accept the tweet {}".format(output))
        return render_template('index.html', prediction_text="Can accept the tweet {}".format(output))
    #return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)