import random

import numpy

from deap import base
from deap import creator
from deap import tools


class RealEA:
    def __init__(self, gpIndividual, fitness_function):
        self.gpIndividual = gpIndividual
        self.fitness_function = fitness_function

        # extract all weights from the individual
        self.initial_weights = self.extractWeights()

        # Single objective Fitness class, minimizing fitness
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Individual class with base type list with a fitness attribute set to jus tthe created fitness
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Equal to arity of the functions
        NUM_WEIGHTS = len(self.initial_weights)
        POP_SIZE = 10

        # Attribute generator
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)

        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_WEIGHTS)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)

        # Operator registering
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("crossover", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=2)

        self.toolbox = toolbox

    def main(self):
        # initialize the population with random individuals
        pop = self.toolbox.population

        print('Type of pop',type(pop))
        # replace one individual with old weights
        pop[0] = self.initial_weights


        CROSS_PROB, M_PROB, MAX_GEN= 0.5,0.2, 20

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)

        # Keep track of elite individual and its fitness
        elite = []
        elite_fitness = 1
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            if fit < elite_fitness:
                elite = ind
                elite_fitness = fit


        for g in range(MAX_GEN):
            # Select the next generation individuals
            selected = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = map(self.toolbox.clone, selected)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSS_PROB:
                    self.toolbox.crossover(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < M_PROB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.fitness.values = fit

                # compare with current elite
                if fit < elite_fitness:
                    elite = ind
                    elite_fitness = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        return elite


    # Evaluation function
    def evaluate(self, individual):
        return self.fitness_function.getFitness(individual)


    def extractWeights(self):
        weights = []
        subtree = self.gpIndividual.GetSubtree()

        # Add original weights to the list
        for index in range(len(subtree)):
            weights.append(subtree[index].w0)
            weights.append(subtree[index].w1)

        return weights



# TODO: Use HallOfFame?
