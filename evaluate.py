import numpy as np
import pandas as pd
import math

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
    features = data[data.columns[:-1]]
    results = data[data.columns[-1]]
    mean = results.mean()
    ss_res = 0
    ss_tot = 0
    for index, row in features.iterrows():
        ss_res += (results[index] - func(*row))**2
        ss_tot += (results[index] - mean)**2
    return 1 - (ss_res/ss_tot),
    