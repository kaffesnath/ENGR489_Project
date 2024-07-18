import operator
import numpy as np
import pandas as pd
import random
import preprocess
from gensim.models import Word2Vec as w2v
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer;
import sys
import re

def clean_data(text):
    #remove links
    text = str(text)
    text = re.sub('(https?:\/\/)?(\w+\.)?(\w+\.)?\w+\.\w+(\s*\/\s*\w+)*', '', text) 
    #remove mentions or user handles
    text = re.sub('(^|\B)@ \w+', '', text)
    text = re.sub('(^|\B)@\w+', '', text)
    #remove special characters
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    text = re.sub(r' +', ' ', text).strip()
    text = remove_stopwords(text)
    return text

def replace_sentiment(sentiment):
    sentiment = sentiment.lower()
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return 0
    else:
        return 2

def get_data():
    data = pd.read_csv(sys.argv[1], names=['tweet_id', 'entity', 'sentiment', 'tweet_content'])
    data = data[['tweet_content', 'sentiment']]
    data['sentiment'] = data['sentiment'].apply(replace_sentiment)

    sentences = []
    corpus = []
    embeddings = {}
    
    dimensions = 100

    #preprocess and clean data
    for index, row in data.iterrows():
        sentence = clean_data(row['tweet_content'])
        sentences.append(sentence)
        corpus.append(sentence.split())

    #Window represents the field in which relations are calcuated and considered for the matrix
    #Min count represents the lowest repetitions required for a word to be considered for the matrix
    #Workers represents the number of threads used to train the model
    w2v_model = w2v(corpus, vector_size=dimensions, window=5, min_count=1, workers=4)

    word_set = set()
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            word_set.add(word)
    
    for word in word_set:
        if word in w2v_model.wv:
            embeddings[word] = w2v_model.wv[word]
        else:
            embeddings[word] = np.zeros(dimensions)
    
    sentence_data = []
    for index, row in data.iterrows():
        sentence = str(row['tweet_content'])
        sentence_embedding = np.zeros(dimensions)
        for word in sentence.split():
            if word in embeddings:
                sentence_embedding += embeddings[word]
        sentence_data.append(sentence_embedding)
    #convert to dataframe
    sentence_data = pd.DataFrame(sentence_data)
    sentence_data[dimensions] = data['sentiment']
    return sentence_data
            
    



