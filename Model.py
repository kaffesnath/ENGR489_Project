import operator
import numpy as np
import pandas as pd
import random
import preprocess
import evaluate
from deap import base, creator, gp, tools, algorithms
from sklearn.feature_extraction.text import TfidfVectorizer;

toolbox = base.Toolbox()

data = preprocess.get_data()

def eval(individual):
    print(data)
    points = points=[x/10. for x in range(-10,10)]
    func = toolbox.compile(expr=individual)
    return evaluate.evaluate_model(func, points)

def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1

#define pset
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
#terminal set 0 or 1
pset.addTerminal(1)
pset.addTerminal(0)
pset.renameArguments(ARG0='x')

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
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof

if __name__ == "__main__":
    main()
