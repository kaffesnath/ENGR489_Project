import numpy as np
import pandas as pd
from gensim.models import Word2Vec as w2v
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Doc2Vec as d2v
from gensim.models.doc2vec import TaggedDocument
import sys
import re

def clean_data(text):
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
    text = PorterStemmer().stem(text)
    text = WordNetLemmatizer().lemmatize(text)
    if text == '':
        return 'empty'
    return text

def replace_sentiment(sentiment):
    sentiment = sentiment.lower()
    if sentiment == 'positive':
        return 2
    elif sentiment == 'negative':
        return -2
    else:
        return 0

def get_data():
    data = pd.read_csv('datasets/' + sys.argv[1], names=['tweet_id', 'entity', 'sentiment', 'tweet_content'])
    data = data[['tweet_content', 'sentiment']]
    data['sentiment'] = data['sentiment'].apply(replace_sentiment)

    corpus = []
    features = []
    
    dimensions = 50

    #preprocess and clean data
    for index, row in data.iterrows():
        sentence = clean_data(row['tweet_content'])
        if sentence != 'empty':
            corpus.append(TaggedDocument(sentence.split(), [str(index)]))


    # Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. 
    # Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
    
    sia = SentimentIntensityAnalyzer()
    d2v_model = d2v(corpus, vector_size=dimensions, window=5, min_count=1, workers=4)

    for sentence in corpus:
        d2v_embed = d2v_model.infer_vector(sentence.words)
        sia_embed = sia.polarity_scores(' '.join(sentence.words))
        features.append(np.concatenate((d2v_embed, list(sia_embed.values()))))
    features = pd.DataFrame(features)
    features[len(features.columns)] = data['sentiment']
    return features

def main():
    features = get_data()
    features.to_csv('datasets/features.csv', index=False)
            
    



