import operator
import numpy as np
import pandas as pd
import random
import preprocess
import evaluate
from deap import base, creator, gp, tools, algorithms
from sklearn.feature_extraction.text import TfidfVectorizer;

toolbox = base.Toolbox()

#Split ratio constant
SPLIT_CONSTANT = 0.8

data, results = preprocess.get_data()
split = int(len(data) * SPLIT_CONSTANT)
test = data.tail(len(data)-split)
test_results = results[split:]
data = data.head(split)

def eval(individual):
    func = toolbox.compile(expr=individual)
    return evaluate.evaluate_model(func, test, test_results)

def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1
    
#define pset
pset = gp.PrimitiveSet("MAIN", len(data.columns))
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
#define terminal set
pset.addTerminal(random.uniform(-1, 1))
#argument definition
for i in range(0, len(data.columns)):
    pset.renameArguments(ARG0='x{}'.format(i))


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
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
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 50, stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof

if __name__ == "__main__":
    main()
