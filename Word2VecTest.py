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
wnl = WordNetLemmatizer()

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
    #text = PorterStemmer().stem(text)
    text = ' '.join([wnl.lemmatize(word) for word in text.split()])
    if text == '':
        return 'empty'
    return text

def create_sentiment(x):
    nums = x.split(',')
    #drop first item
    nums = nums[1:]
    #convert to integers
    nums = [int(i) for i in nums]
    #return average of list
    return sum(nums)/len(nums)

def get_data():
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
    corpus = []
    features = []
    
    dimensions = 50

    #preprocess and clean data
    for index, row in data.iterrows():
        sentence = clean_data(row['tweet_content'])
        corpus.append(sentence)

    #map corpus data to main dataset
    data['tweet_content'] = corpus

    #drop all with empty string
    data = data[data['tweet_content'] != 'empty']

    #drop duplicates
    data.drop_duplicates(subset='tweet_content', inplace=True)
    corpus = data['tweet_content'].tolist() 
    #tag each sentence with an index
    corpus = [TaggedDocument(words=sentence.split(), tags=[str(i)]) for i, sentence in enumerate(corpus)]

    # Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. 
    # Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
    
    sia = SentimentIntensityAnalyzer()
    d2v_model = d2v(corpus, vector_size=dimensions, window=5, min_count=1, workers=4)

    for sentence in corpus:
        d2v_embed = d2v_model.infer_vector(sentence.words)
        sia_embed = sia.polarity_scores(' '.join(sentence.words))
        features.append(np.concatenate((d2v_embed, list(sia_embed.values()))))
    features = pd.DataFrame(features)
    data.reset_index(drop=True, inplace=True)
    features[len(features.columns)] = data['sentiment']
    #take first 1% of data for testing after shuffling
    features = features.sample(frac=1)
    features = features[:int(len(features) * 0.01)]
    return features

def main():
    features = get_data()
    features.to_csv('datasets/features.csv', index=False)
            
    



