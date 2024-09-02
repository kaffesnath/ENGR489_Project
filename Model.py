import operator
import numpy as np
import pandas as pd
import random
import evaluate
import warnings
import sys
from deap import base, creator, gp, tools, algorithms

warnings.filterwarnings("ignore")
toolbox = base.Toolbox()

#Get data
data = pd.read_csv('datasets/features,csv')

def eval(individual):
    func = toolbox.compile(expr=individual)
    return evaluate.evaluate_model(data, func),

def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1
    
def main():
    #all parameters
    MAX_DEPTH = 8
    TOURNEY_SIZE = 10
    CXPB = 0.85
    MUTPB = 0.15
    NGEN = 50
    POP = 300

    #define pset
    pset = gp.PrimitiveSet("MAIN", len(data.columns) - 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    # Define genetic operators
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=TOURNEY_SIZE)
    toolbox.register("evaluate", eval)

    random.seed(999)
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
main()
