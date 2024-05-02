import operator
import numpy as np
import pandas as pd
import random
import preprocess
from deap import base, creator, gp, tools, algorithms

def evaluate_model(individual, data, toolbox, target):
    func = toolbox.compile(expr=individual)
    return np.mean(np.abs(func(data) - target)),