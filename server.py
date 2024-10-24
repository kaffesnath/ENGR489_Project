from flask import render_template, Flask, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Doc2Vec
from Pipeline import clean_data
from sklearn.preprocessing import StandardScaler
import pickle
from deap import base, creator, gp, tools, algorithms
import operator
import numpy as np
ss = StandardScaler()

def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1

def create_pset():
    pset = gp.PrimitiveSet("MAIN", 5)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addTerminal(2.5)
    return pset

def calc_sentiment(text):
    # Load models for calculating sentiment
    d2v_model = Doc2Vec.load('datasets/stanford/d2v.model')
    sia_model = SentimentIntensityAnalyzer()
    lsvc_model = pickle.load(open('datasets/stanford/lsvc.sav', 'rb'))
    func = pickle.load(open('datasets/stanford/func.sav', 'rb'))

    # Clean the text
    text = clean_data(text)
    if text == ('empty', 'empty'):
        return "Input is too short"
    text = text[0]
    print(text)
    # Infer the vector from d2v pretrained model and the vader values
    vectors = d2v_model.infer_vector(text.split())
    sia_embed = list(sia_model.polarity_scores(text).values())
    #concatenate the two vectors
    vectors = [np.concatenate((vectors, sia_embed))]
    estimate = lsvc_model.predict(vectors)
    #create features from the estimate
    features = np.concatenate((sia_embed, estimate))
    #calculate sentiment score
    pset = create_pset()
    print(features)
    func = gp.compile(expr=func, pset=pset)
    polarity_score = func(*features)
    # Return the sentiment
    return polarity_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def output():
    string = request.form['input']
    return render_template('index.html', result=calc_sentiment(string))

if __name__ == '__main__':
    app.run()