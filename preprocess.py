from sklearn.feature_extraction.text import TfidfVectorizer;
import pandas as pd
import sys
import numpy as np
import re

data = pd.read_csv(sys.argv[1], names=['tweet_id', 'entity', 'sentiment', 'tweet_content'])

tfidf = TfidfVectorizer()
def preprocess(description):
    tfidf.fit_transform(description)

def print_features():
    for el1, el2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
        print(el1, " : ", el2)

def longest_sentence(data):
    max = 0
    for index, row in data.iterrows():
        sentence = len(row['tweet_content'].split())
        if sentence > max:
            max = sentence
    return max

def get_features():
    features = np.zeros((len(data), longest_sentence(data)))
    for index, row in data.iterrows():
        description = row['tweet_content']
        response = []
        for word in description.split():
            if word in tfidf.get_feature_names_out():
                response.append(tfidf.idf_[tfidf.vocabulary_[word]])
        for i in range(0, len(response)):
            features[index][i] = response[i]
    return pd.DataFrame(features)

def clean_data(text):
    #remove links
    text = str(text)
    text = re.sub(r'pic\.twitter\.com/.*', "", text)
    text = re.sub(r'pic.twitter.com/[\w]*',"", text)
    text = re.sub(r'dlvr.it /[\w]*',"", text)
    text = re.sub(r'dlvr.it/[\w]*',"", text)
    text = re.sub(r'twitch.tv / [\w]*',"", text)
    text = re.sub(r'twitch.tv/[\w]*',"", text)
    #remove special characters
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

def get_data():
    results = np.zeros(len(data))
    for index, row in data.iterrows():
        row['sentiment'] = row['sentiment'].lower()
        row['tweet_content'] = clean_data(row['tweet_content'])
        #replace sentiment with numerical values
        if row['sentiment'] == 'positive':
            data.at[index, 'sentiment'] = 1
        elif row['sentiment'] == 'negative':
            data.at[index, 'sentiment'] = 0
        else:
            data.at[index, 'sentiment'] = 2
        data.at[index, 'tweet_content'] = row['tweet_content']
    corpus = []
    for index, row in data.iterrows():
        #preprocess the tweet content
        if(data.at[index, 'sentiment'] != 2):
            corpus.append(data.at[index, 'tweet_content'])
            results[index] = data.at[index, 'sentiment']
    preprocess(corpus)
    features = get_features()
    return features, results