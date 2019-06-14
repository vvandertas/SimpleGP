from scipy.io import arff
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

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


# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
sgp = SimpleGP(fitness_function, functions, terminals, pop_size=100, max_generations=100
    ,weight_tuning_individual_rate=1.0,weight_tuning_generation_rate=5,weight_tuning_max_generations=20)  # other parameters are optional
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
print('Test:\n\terror rate:', test_error_rate)
