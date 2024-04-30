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
        wordCount = 0
        averageTfidf = 0
        description = row['tweet_content']
        words = description.split()
        for word in words:
            if word in tfidf.get_feature_names_out():
                wordCount += 1
                averageTfidf += tfidf.idf_[tfidf.vocabulary_[word]]
        if wordCount != 0:
            averageTfidf /= wordCount
        row = [row['sentiment'], wordCount, averageTfidf]
        features.append(row)
    return pd.DataFrame(features)

def main():
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
    print(features)
main()