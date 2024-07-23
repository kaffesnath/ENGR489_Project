import numpy as np
import pandas as pd

def evaluate_model(data, func):
    score = 0
    for index, row in data.iterrows():
        inst = row[:row.size - 1]
        sentiment = row[row.size - 1]
        #set upper and lower bounds for sentiment
        upper = 0.8
        lower = 0.2
        result = func(*inst)
        if sentiment == 1 and result > upper:
            score += 1
        elif sentiment == 0 and result < lower:
            score += 1
        elif sentiment == 2 and result >= lower and result <= upper:
            score += 1
    return score / len(data)
            
    