from scipy.io import arff
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

import numpy as np
import pandas as pd
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP

# Load dataset
data, meta = arff.loadarff('data/cpu_act.arff')
df = pd.DataFrame(data)

# target variable
y = df.binaryClass.values
print(type(y))

# other data
X = df.drop('binaryClass', axis=1).values #df.iloc[:, df.columns != 'binaryClass'].values
print(type(X))

result_data_folder = "result_data/"

GP_POP_SIZE = 100
GP_MAX_GENERATIONS = 100
WEIGHT_TUNING_INDIVIDUAL_RATE = 0.0 #1.0
WEIGHT_TUNING_GENERATION_RATE = -1 #5
WEIGHT_TUNING_MAX_GENERATIONS = 0 #20
REAL_POP_SIZE = None #10
REAL_CROSSOVER_RATE = None #0.5
REAL_MUTATION_RATE = None #0.5


parameters = str(GP_POP_SIZE) + "_" + str(GP_MAX_GENERATIONS) + "_" + str(WEIGHT_TUNING_INDIVIDUAL_RATE) + "_" + str(WEIGHT_TUNING_GENERATION_RATE) + "_" + str(WEIGHT_TUNING_MAX_GENERATIONS) + "_" + str(REAL_POP_SIZE) + "_" + str(REAL_CROSSOVER_RATE) + "_" + str(REAL_MUTATION_RATE)

filename = result_data_folder + parameters + ".csv"

global_result_filename = result_data_folder + "global_results.csv"

test_errors = []
nr_of_evaluations = []
tree_sizes = []
best_test_error = 1.0


file = open(filename,"w+")
file.write("test error,nr of evaluations, tree size")

# create training and testing vars
rkf = RepeatedKFold(n_splits=10, n_repeats=10,random_state=42)
for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    print('Type of X_train: ', X_train.dtype)
    print('Type of y_train: ', y_train.dtype)

    # Set fitness function
    fitness_function = SymbolicRegressionFitness(X_train, y_train.astype(float))
    print(fitness_function)


    # Set functions and terminals
    functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
    terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node
    for i in range(X.shape[1]):
        terminals.append(FeatureNode(i))  # add a feature node for each feature

    # Run GP
    sgp = SimpleGP(fitness_function, functions, terminals, pop_size=GP_POP_SIZE, max_generations=GP_MAX_GENERATIONS
        ,weight_tuning_individual_rate=WEIGHT_TUNING_INDIVIDUAL_RATE,weight_tuning_generation_rate=WEIGHT_TUNING_GENERATION_RATE
        ,weight_tuning_max_generations=WEIGHT_TUNING_MAX_GENERATIONS
        ,real_pop_size=REAL_POP_SIZE,real_crossover_rate=REAL_CROSSOVER_RATE,real_mutation_rate=REAL_MUTATION_RATE)  # other parameters are optional
    sgp.Run()

    # Print results
    # Show the evolved function
    final_evolved_function = fitness_function.elite
    nodes_final_evolved_function = final_evolved_function.GetSubtree()
    print('Function found (', len(nodes_final_evolved_function), 'nodes ):\n\t',
          nodes_final_evolved_function)  # this is in Polish notation

    # Print results for training set
    print('Training\n\terror rate:', final_evolved_function.fitness)

    # Re-evaluate the evolved function on the test set
    test_prediction = np.arctan(final_evolved_function.GetOutput(X_test))
    test_prediction[test_prediction >= 0] = 1
    test_prediction[test_prediction < 0] = -1
    test_error_rate = 1 - (np.sum(y_test == test_prediction) / len(y_test))
    evaluation_count = sgp.fitness_function.evaluations
    tree_size = len(nodes_final_evolved_function)
    print('Test:\n\terror rate:', test_error_rate)
    test_errors.append(test_error_rate)
    nr_of_evaluations.append(evaluation_count)
    tree_sizes.append(tree_size)
    file.write("\n" + str(test_error_rate) + "," + str(evaluation_count) + "," + str(tree_size))
    
    if test_error_rate < best_test_error:
        best_test_error = test_error_rate
        
file.close()

result = parameters+","+str(np.mean(test_errors))+","+str(np.mean(nr_of_evaluations))+","+str(np.mean(tree_sizes))+","+str(best_test_error)
global_file = open(global_result_filename,"a")
global_file.write("\n" + result)
global_file.close

