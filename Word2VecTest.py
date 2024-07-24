import numpy as np
import pandas as pd
from gensim.models import Word2Vec as w2v
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
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
        return 1
    elif sentiment == 'negative':
        return 0
    else:
        return 2

def get_data():
    data = pd.read_csv(sys.argv[1], names=['tweet_id', 'entity', 'sentiment', 'tweet_content'])
    data = data[['tweet_content', 'sentiment']]
    data['sentiment'] = data['sentiment'].apply(replace_sentiment)

    corpus = []
    embeddings = []
    
    dimensions = 40

    #preprocess and clean data
    for index, row in data.iterrows():
        sentence = clean_data(row['tweet_content'])
        if sentence != 'empty':
            corpus.append(TaggedDocument(sentence.split(), [str(index)]))

    #Window represents the field in which relations are calcuated and considered for the matrix
    #Min count represents the lowest repetitions required for a word to be considered for the matrix
    #Workers represents the number of threads used to train the model
    d2v_model = d2v(corpus, vector_size=dimensions, window=5, min_count=1, workers=4)

    for sentence in corpus:
        embeddings.append(d2v_model.infer_vector(sentence.words))
    embeddings = pd.DataFrame(embeddings)
    embeddings[len(embeddings.columns)] = data['sentiment']
    return embeddings
            
    



