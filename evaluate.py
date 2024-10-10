import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score

def evaluate_model_mse(data, func):
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

def evaluate_model_rmse(data, func):
    return math.sqrt(evaluate_model_mse(data, func)[0]),

def evaluate_model_r_squared(data, func):
    return r2_score(data[data.columns[-1]], [func(*row) for index, row in data[data.columns[:-1]].iterrows()]),
    