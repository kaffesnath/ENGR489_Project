import operator
import numpy as np
import pandas as pd
import random
import Word2VecTest
import evaluate
from sklearn.ensemble import RandomForestClassifier
from deap import base, creator, gp, tools, algorithms

toolbox = base.Toolbox()

#Get data
data = Word2VecTest.get_data()

def eval(individual):
    func = toolbox.compile(expr=individual)
    return evaluate.evaluate_model(data, func),

def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1
    
#define pset
pset = gp.PrimitiveSet("MAIN", len(data.columns) - 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
#define terminal set
pset.addTerminal(random.uniform(-1, 1))
#argument definition
for i in range(0, len(data.columns) - 1):
    pset.renameArguments(ARG0='x{}'.format(i))


creator.create("FitnessMax", base.Fitness, weights=(0.5,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
# Define genetic operators
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", eval)

def main():
    random.seed(999)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(2)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 100, stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof
main()
