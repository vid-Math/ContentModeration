import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from LSVC_util import *

import keras 
from keras.utils import pad_sequences

def model_LSVC_predict(tweet):
    filename = 'model_LSVC.pk'
    with open('./'+filename ,'rb') as f:
        model_LSVC = pickle.load(f)
    
    filename = 'vectoriser_LSVC.pk'
    with open('./'+filename ,'rb') as f:
        vectorizer = pickle.load(f)
    #print(vectorizer.get_feature_names_out())

    
    X_tweet = vectorizer.transform(tweet)
    prediction = model_LSVC.predict(X_tweet)
    return prediction

def model_LSTM_predict(tweet):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    list_tokenized_test = tokenizer.texts_to_sequences(tweet)
    maxlen = 200
    X = pad_sequences(list_tokenized_test, maxlen=maxlen)


    filename = 'model_lstm.keras'
    model_LSTM = keras.models.load_model(filename)
    prediction = model_LSTM.predict(X)
    print("===prediction===", prediction)

    if prediction<0.5:
        prediction = 0
    elif prediction>0.5:
        prediction = 1
    return prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    tweet_ip =  request.form.get('tweet')
    tweet = []
    tweet.append(request.form.get('tweet'))
    print("tweet---- ", tweet)
    
    model_option = request.form.get('model_option')
    print("model selection ---- ", model_option)
    #print("model selection type ---- ", type(model_option))
    if model_option=='lsvc':
        prediction = model_LSVC_predict(tweet)
    elif model_option=='lstm':
        prediction = model_LSTM_predict(tweet)

    output = prediction
    print("tyep(output)====> ", output)

    if output==1:
        print("Can't accept the tweet {}".format(output))
        return render_template('index.html', prediction_text="Can't accept the tweet")#.format(output))
    else:
        print("Can accept the tweet {}".format(output))
        return render_template('index.html', prediction_text="Acceptable!".format(output))
    #return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     print("=====",data,"======")
#     prediction = model_LSVC_predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == '__main__':
    app.run(port=5002, debug=True)