import Word2VecTest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

SPLIT_CONSTANT = 0.8
def split_data(data, x):
    train, test = train_test_split(data, test_size=1 - SPLIT_CONSTANT, random_state=x)
    #set train
    train_results = train[train.columns[-1]]
    train = train.drop(train.columns[-1], axis=1)
    #set test
    test_results = test[test.columns[-1]]
    test = test.drop(test.columns[-1], axis=1)
    return train, test, train_results, test_results

def main():
    data = Word2VecTest.get_data()
    training_data = []
    #train model
    model = DecisionTreeClassifier()
    for i in range(0, 50):
        train, test, train_results, test_results = split_data(data, i)
        model.fit(train, train_results)
        #test model
        training_data.append(sum(model.predict(test) == test_results) / len(test))
    print("Accuracy:", np.mean(training_data))

main()
