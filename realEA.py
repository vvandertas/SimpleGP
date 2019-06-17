import random
from copy import deepcopy

from deap import base
from deap import creator
from deap import tools


class RealEA:
    def __init__(
            self,
            gp_individual,
            fitness_function,
            pop_size=10,
            crossover_rate=0.5,
            mutation_rate=0.5,
            max_generations=20):

        
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

        self.gp_individual = gp_individual
        self.fitness_function = fitness_function

        # extract all weights from the individual
        self.initial_weights = self.extract_weights()

        # Single objective Fitness class, minimizing fitness
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Individual class with base type list with a fitness attribute set to just the created fitness
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Based on the number of weights in the current individual (depends on tree size)
        NUM_WEIGHTS = len(self.initial_weights)

        # Attribute generator
        toolbox = base.Toolbox()
        toolbox.register("scalar_weight", random.gauss,mu=1,sigma=0.5)
        toolbox.register("bias_weight",random.gauss,mu=0,sigma=0.5)

        # Structure initializers
        toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.bias_weight, toolbox.scalar_weight), n=int(NUM_WEIGHTS/2))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)

        # Operator registering
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("crossover", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=2)

        self.toolbox = toolbox

    def main(self):
        # initialize the population with random individuals
        pop = self.toolbox.population()

        # replace one individual with old weights
        pop[0].n = self.initial_weights

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)

        # Keep track of elite individual and its fitness
        elite = self.initial_weights
        elite_fitness = self.evaluate(self.initial_weights)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            if fit[0] < elite_fitness[0]:
                elite = deepcopy(ind)
                elite_fitness = deepcopy(fit)

        for g in range(self.max_generations):
            # Select the next generation individuals
            selected = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, selected))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.crossover(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # and mutation on the offspring
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

                # compare with current elite
                if fit[0] < elite_fitness[0]:
                    elite = deepcopy(ind)
                    elite_fitness = deepcopy(fit)

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        # Only return the elite
        return elite

    # Evaluation function
    def evaluate(self, individual):
        cloned_tree = deepcopy(self.gp_individual)
        cloned_tree.set_weights(individual)

        return self.fitness_function.getFitness(cloned_tree),

    def extract_weights(self):

        weights = []
        subtree = self.gp_individual.GetSubtree()

        # Add original weights to the list
        for index in range(len(subtree)):
            weights.append(subtree[index].w0)
            weights.append(subtree[index].w1)

        return weights
