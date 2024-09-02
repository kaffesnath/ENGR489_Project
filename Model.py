import operator
import numpy as np
import pandas as pd
import random
import evaluate
import warnings
import sys
import Word2VecTest as pps
from deap import base, creator, gp, tools, algorithms

warnings.filterwarnings("ignore")
toolbox = base.Toolbox()

#Get data
data = pps.get_data()
def eval(individual):
    func = toolbox.compile(expr=individual)
    return evaluate.evaluate_model(data, func)

def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1
    
def protectedSqrt(a):
    if a < 0:
        return 1
    else:
        return np.sqrt(a)

def main(seed):
    #all parameters
    random.seed(seed)
    MAX_DEPTH = 10
    TOURNEY_SIZE = 20
    CXPB = 0.7
    MUTPB = 0.3
    NGEN = 50
    POP = 400

    #define pset
    pset = gp.PrimitiveSet("MAIN", len(data.columns) - 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addEphemeralConstant("rand101", lambda: random.randint(-2,2))

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
pop, log, hof = main(999)
print(hof[0])
print(hof[0].fitness.values)
