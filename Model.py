import operator
import numpy as np
import pandas as pd
import random
from evaluate import evaluate_model_mse as mse
from evaluate import evaluate_model_rmse as rmse
from evaluate import evaluate_model_r_squared as r_squared
from sklearn.model_selection import train_test_split
import warnings
import sys
import multiprocessing
import Pipeline as pps
import matplotlib.pyplot as plt
from deap import base, creator, gp, tools, algorithms
import pickle

warnings.filterwarnings("ignore")
toolbox = base.Toolbox()

#Get data
data = pps.get_data()
#data = data.sample(frac=0.001)
train, test = train_test_split(data, test_size=0.1, random_state=42)
fitness = 'mse'
test_type = 'class'

def eval(individual):
    # Transform the tree expression in a callable function
    try:
        func = toolbox.compile(expr=individual)
    except SyntaxError:
        return sys.float_info.max,
    # Evaluate the mean squared error between the expression
    # and the real function MSE
    if fitness == 'mse':
        return mse(data, func)
    elif fitness == 'rmse':
        return rmse(data, func)
    else:
        return r_squared(data, func)

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
    pset.addPrimitive(operator.neg, 1)
    pset.addTerminal(2.5)
    if fitness == 'r2':
        creator.create("Fitness", base.Fitness, weights=(1.0,))
    else:
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, pset=pset)

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
    if fitness == 'r2':
        return log.chapters['fitness'].select('max'), hof
    return log.chapters['fitness'].select('min'), hof

def run_gp():
    num_threads = 4
    num_runs = 10
    with multiprocessing.Pool(num_threads) as pool:
        results = pool.map(gp_instance, range(num_runs))
    pool.close()
    pool.join()
    gens_total, hof = zip(*results)
    gens_total = list(gens_total)
    hof = list(hof)
    print(hof[-1][0])
    return gens_total


def test_fitness():
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
    plt.title('Average {} over 10 runs'.format(fitness.upper()))
    plt.ylabel('Average {}'.format(fitness.upper()))
    plt.xlabel('Generation')
    plt.show()

    print('Average Ending {}:{}'.format(fitness.upper(), np.mean(final)))
    print('Standard Deviation Ending {}:{}'.format(fitness.upper(), np.std(final)))
    if fitness == 'r2':
        print('Maximum Ending {}:{}'.format(fitness.upper(), np.max(final)))
    else:
        print('Minimum Ending {}:{}'.format(fitness.upper(), np.min(final)))

def class_eval():
    log, hof = gp_instance(random.randint(2, 100))
    func = hof[0]
    print('Best Individual:', func)
    pickle.dump(func, open('datasets/stanford/func.sav', 'wb'))
    func = toolbox.compile(expr=func)
    print('Training Accuracy:', log[-1])
    print('Test Accuracy:', mse(test, func)[0])


def main(args):
    global data
    if args == 'test':
        test_fitness()
    else:
        data = train
        class_eval()

if __name__ == '__main__':
    main(test_type)

