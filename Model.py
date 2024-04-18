import operator
import numpy as np
import pandas as pd
import sys
import random
from deap import base, creator, gp, tools, algorithms
from sklearn.feature_extraction.text import TfidfVectorizer;

tfidf = TfidfVectorizer()
def preprocess(description):
    tfidf.fit_transform(description)

def evaluate(x):
    # Load data
    data = pd.read_csv(sys.argv[1])
    # Preprocess data
    vectorised = []
    for row in data.iterrows():
        description = row[3]
        preprocess(description)
        newRow = [tfidf.transform(description), 1 if row[2] == 'Positive' else 0]
        vectorised.append(newRow)
    # Split data
    x = [row[0] for row in vectorised]
    y = [row[1] for row in vectorised]
    # Train model
    model = RandomForestClassifier()
    model.fit(x, y)
    # Evaluate model
    return model.score(x, y)

def RandomForestClassifier():
    return 0.5

toolbox = base.Toolbox()

#define pset
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.truediv, 2)
pset.addTerminal(random.uniform(-1, 1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
# Define genetic operators
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutNodeReplacement)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate)

def main():
    random.seed(318)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof

if __name__ == "__main__":
    main()
