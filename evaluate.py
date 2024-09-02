import numpy as np
import pandas as pd
import math

def evaluate_model(data, func):
    #get the features
    features = data[data.columns[:-1]]
    #get the results
    results = data[data.columns[-1]]
    #get the predictions
    sqerrors = []
    for index, row in features.iterrows():
        sqerrors.append((func(*row) - results[index])**2)
    #return the accuracy
    return math.fsum(sqerrors) / len(sqerrors), 
            
    