import operator
import numpy as np
import pandas as pd
import random
from evaluate import evaluate_model_mse as mse
from evaluate import evaluate_model_rmse as rmse
from evaluate import evaluate_model_r_squared as r_squared
import warnings
import sys
import multiprocessing
import Word2VecTest as pps
import matplotlib.pyplot as plt
from deap import base, creator, gp, tools, algorithms

warnings.filterwarnings("ignore")
toolbox = base.Toolbox()

#Get data
data = pps.get_data()

def eval(individual):
    # Transform the tree expression in a callable function
    try:
        func = toolbox.compile(expr=individual)
    except SyntaxError:
        return sys.float_info.max,
    # Evaluate the mean squared error between the expression
    # and the real function MSE
    return rmse(data, func)

def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1

def create_toolbox():
    MAX_DEPTH = 6
    TOURNEY_SIZE = 10

    pset = gp.PrimitiveSet("MAIN", len(data.columns) - 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addEphemeralConstant("x1", lambda: random.uniform(0.1, 2))
    pset.addEphemeralConstant("x2", lambda: random.uniform(-2, -0.1))

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

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))

def gp_instance(seed):
    #all parameters
    create_toolbox()
    global toolbox
    random.seed(seed * 8)
    CXPB = 0.7
    MUTPB = 0.3
    NGEN = 50
    POP = 100
    
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
    return log.chapters['fitness'].select('min')

def run_gp():
    num_threads = 4
    num_runs = 10
    with multiprocessing.Pool(num_threads) as pool:
        gens_total = pool.map(gp_instance, range(num_runs))
    pool.close()
    pool.join()
    return gens_total

def main():
    gens_total = run_gp()
    final = []
    for i in range(len(gens_total)):
        final.append(gens_total[i][-1])
    
    #plot the results
    average = []
    for i in range(len(gens_total[0])):
        total = 0
        for j in range(len(gens_total)):
            total += gens_total[j][i]
        average.append(total / len(gens_total))
    plt.plot(average)
    plt.title('Average MSE over 10 runs')
    plt.ylabel('Average MSE')
    plt.xlabel('Generation')
    plt.show()

    print('Average Ending MSE:', np.mean(final))
    print('Standard Deviation Ending MSE:', np.std(final))
    print('Minimum Ending MSE:', np.min(final))

if __name__ == '__main__':
    main()

