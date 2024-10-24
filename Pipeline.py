import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Doc2Vec as d2v
from gensim.models.doc2vec import TaggedDocument
import sys
import gensim.downloader as api
from sklearn.preprocessing import StandardScaler
import pickle
import os.path
import re
wnl = WordNetLemmatizer()
ss = StandardScaler()

def clean_data(text):
    original = text
    #remove links
    text = str(text)
    text = re.sub('(https?:\/\/)?(\w+\.)?(\w+\.)?\w+\.\w+(\s*\/\s*\w+)*', '', text) 
    #remove mentions or user handles
    text = re.sub('(^|\B)@ \w+', '', text)
    text = re.sub('(^|\B)@\w+', '', text)
    text = re.sub('(<(.*?)>)', '', text)
    #remove special characters
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    text = re.sub(r' +', ' ', text).strip()
    #removes stopwords and stems the words
    text = remove_stopwords(text)
    #text = PorterStemmer().stem(text)
    text = ' '.join([wnl.lemmatize(word) for word in text.split()])
    if text == '':
        return 'empty', 'empty'
    #drop all with less than 4 words
    if len(text.split()) < 4:
        return 'empty', 'empty'
    return text, original

def create_sentiment(x):
    nums = x.split(',')
    #drop first item
    nums = nums[1:]
    #convert to integers
    nums = [int(i) for i in nums]
    #return average of list
    return round(sum(nums) / len(nums))

def create_features(vectors, sentiment):
    svc = LinearSVC(C = 0.1, max_iter=10000, class_weight='balanced')
    x = ss.fit_transform(vectors)
    y = round(sentiment % 5) + 1
    svc.fit(x, y)
    preds = svc.predict(vectors)
    pickle.dump(svc, open('datasets/stanford/lsvc.sav', 'wb'))
    return preds

def convert_sentiment(x):
    return (x + 1) * 5 - 2.5

def get_data():
    try:
        query = sys.argv[1]
    except IndexError:
        query = ''
    #checks for saved data, else creates, also checks for explicit train argument
    if os.path.isfile('datasets/stanford/processed_data_2.csv') and query != 'train':
        return pd.read_csv('datasets/stanford/processed_data_2.csv')
    
    with open('datasets/stanford/sentlex_exp12.txt') as f:
        lines = f.readlines()
    #remove index of item
    lines = [i.split(',', 1)[1] for i in lines]
    #remove newline characters
    lines = [i.replace('\n', '') for i in lines]
    data = pd.DataFrame(lines)

    with open('datasets/stanford/rawscores_exp12.txt') as f:
        lines = f.readlines()
    data['sentiment'] = [create_sentiment(i) for i in lines]
    data.columns = ['tweet_content', 'sentiment']
    #filter to 0.1% of data
    #data = data.sample(frac=0.001)

    corpus = []
    originals = []
    features = []

    # Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. 
    # Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
    
    sia = SentimentIntensityAnalyzer()
    
    dimensions = 200

    #preprocess and clean data
    for index, row in data.iterrows():
        sentence, original = clean_data(row['tweet_content'])
        originals.append(original)
        corpus.append(sentence)
    #map corpus data to main dataset
    data['tweet_content'] = corpus
    data['original'] = originals

    #drop all with empty string
    data = data[data['tweet_content'] != 'empty']

    #drop duplicates
    data.drop_duplicates(subset='tweet_content', inplace=True, keep='first')
    corpus = data['tweet_content'].tolist() 
    #tag each sentence with an index
    corpus = [TaggedDocument(words=sentence.split(), tags=[str(i)]) for i, sentence in enumerate(corpus)]

    # load google news doc2vec model
    d2v_model = d2v(corpus, vector_size=dimensions, window=3, min_count=1, workers=4, epochs=20)
    d2v_model.save('datasets/stanford/d2v.model')

    data.reset_index(drop=True, inplace=True)
    for index, row in data.iterrows():
        d2v_embed = d2v_model.infer_vector(corpus[index].words)
        sia_embed = list(sia.polarity_scores(row['original']).values())
        features.append(np.concatenate((d2v_embed, sia_embed)))
    features = pd.DataFrame(features)
    #create prediction label
    features['sentiment_estimate'] = create_features(features, data['sentiment'])
    features[len(features.columns)] = data['sentiment']
    features.drop(features.columns[:dimensions], axis=1, inplace=True)
    export_data(features)
    return features

def export_data(data):
    data.to_csv('datasets/stanford/processed_data_2.csv', index=False)
    

            
    



