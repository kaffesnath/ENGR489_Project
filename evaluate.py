import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer;

def evaluate_model(func, test, test_results):
    correct = 0
    result_counter = 0
    for index, row in test.iterrows():
        if func(*test) > 0.5 and test_results[result_counter] == 1:
            correct += 1
        elif func(*test) <= 0.5 and test_results[result_counter] == 0:
            correct += 1
        result_counter += 1
    return correct / len(test),
    