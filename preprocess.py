from sklearn.feature_extraction.text import TfidfVectorizer;
import pandas as pd
import sys

data = pd.read_csv(sys.argv[1], names=['tweet_id', 'entity', 'sentiment', 'tweet_content'])

tfidf = TfidfVectorizer()
def preprocess(description):
    tfidf.fit_transform(description)

def print_features():
    for el1, el2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
        print(el1, " : ", el2)

def get_features():
    features = []
    for index, row in data.iterrows():
        description = row['tweet_content']
        words = description.split()
        wordValues = []
        for word in words:
            if word in tfidf.get_feature_names_out():
                wordValues.append(tfidf.idf_[tfidf.vocabulary_[word]])
        averageTfidf = sum(wordValues) / len(wordValues)
        maxTfidf = max(wordValues)
        row = [maxTfidf, averageTfidf, row['sentiment']]
        features.append(row)
    return pd.DataFrame(features)

def get_data():
    data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
    for index, row in data.iterrows():
        #replace sentiment with numerical values
        if row['sentiment'] == 'Positive':
            data.at[index, 'sentiment'] = 1
        elif row['sentiment'] == 'Negative':
            data.at[index, 'sentiment'] = 0
        else:
            data.at[index, 'sentiment'] = 2
    corpus = []
    for index, row in data.iterrows():
        #preprocess the tweet content
        corpus.append(data.at[index, 'tweet_content'])
    preprocess(corpus)
    features = get_features()
    return features