import operator
import numpy as np
import pandas as pd
import random
import preprocess
from deap import base, creator, gp, tools, algorithms

def evaluate_model(func, points):
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return np.sum(sqerrors) / len(points)