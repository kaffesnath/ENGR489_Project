import operator
import numpy as np
import pandas as pd
import random
from evaluate import evaluate_model_mse as mse
from evaluate import evaluate_model_rmse as rmse
from evaluate import evaluate_model_r_squared as r_squared
import warnings
import sys
import Word2VecTest as pps
import matplotlib.pyplot as plt
from deap import base, creator, gp, tools, algorithms

warnings.filterwarnings("ignore")
toolbox = base.Toolbox()

#Get data
data = pps.get_data()

def eval(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function MSE
    return mse(data, func)

def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1

def main(seed):
    #all parameters
    random.seed(seed)
    MAX_DEPTH = 8
    TOURNEY_SIZE = 20
    CXPB = 0.8
    MUTPB = 0.2
    NGEN = 100
    POP = 400

    #define pset
    pset = gp.PrimitiveSet("MAIN", len(data.columns) - 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

    toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    # Define genetic operators
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=TOURNEY_SIZE)
    toolbox.register("evaluate", eval)

    
    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof
gens_total = []
for i in range(10):
    pop, log, hof = main(i * 8)
    #get every best value for each generation
    gens_total.append(log.chapters['fitness'].select('min'))

#plot the results
average = []
for i in range(len(gens_total[0])):
    total = 0
    for j in range(len(gens_total)):
        total += gens_total[j][i]
    average.append(total / len(gens_total))
plt.plot(average)
plt.title('Average RMSE over 10 runs')
plt.ylabel('Average RMSE')
plt.xlabel('Generation')
plt.show()


