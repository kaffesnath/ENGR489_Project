import preprocess
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

SPLIT_CONSTANT = 0.8

def main():
    data, results = preprocess.get_data()
    split = int(len(data) * SPLIT_CONSTANT)
    #test data
    test = data.tail(len(data)-split)
    test_results = results[split:]
    #train data
    data = data.head(split)
    data_results = results[:split]

    #train model
    model = DecisionTreeClassifier()
    model.fit(data, data_results)
    #test model
    test_predictions = model.predict(test)
    print("Accuracy:", metrics.accuracy_score(test_results, test_predictions))

main()
