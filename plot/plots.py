import csv
import os
import re

import numpy as np

from plot.plotter import Plotter

# Indexes of the meaning of the parameters
GP_POP_SIZE = 0
GP_MAX_GENERATIONS = 1
GP_CROSSOVER_RATE = 2
GP_MUTATION_RATE = 3
WEIGHT_TUNING_INDIVIDUAL_RATE = 4
WEIGHT_TUNING_GENERATION_RATE = 5
WEIGHT_TUNING_MAX_GENERATIONS = 6
REAL_POP_SIZE = 7
REAL_CROSSOVER_RATE = 8
REAL_MUTATION_RATE = 9


def gp_only():
    """
    Plots the tree size vs the mean test error of the gp only runs.

    :return:
    """
    plotter = Plotter('GP only tree size vs test error', 'Tree size', 'Test error')
    directory = '../result_data/gp_only/'
    gp_only_files = os.listdir(directory)
    for filename in gp_only_files:
        plotter.add_trace_from_csv(os.path.join(directory, filename), 'tree size', 'test error')
    plotter.plot()


def plot_global_results():
    data = {}
    with open('../result_data/global_results.csv') as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for line in reader:
            parameters = line['parameters']
            match = re.match(r'100_100_1\.0_0\.5_(.+)_5_100_25_(.+)_(.+)', parameters)
            if not match:
                continue
            individual_rate = match.group(2)
            real_crossover_rate = match.group(2)
            real_mutation_rate = match.group(3)
            name = 'real_crossover_rate={} individual rate={}'.format(real_crossover_rate, individual_rate)
            if name not in data:
                data[name] = {}
            data[name][real_mutation_rate] = line['mean test error']
    plotter = Plotter('Effect of crossover and mutation rate on mean test error', 'Mutation rate', 'Mean test error')
    for name in data:
        x = []
        y = []
        for mutation_rate in sorted(data[name]):
            x.append(mutation_rate)
            y.append(data[name][mutation_rate])
        plotter.add_line(name, x, y)
    plotter.plot()


def gp_only_bars_with_error():
    bar_for_crossover = {
        '0.1': {
            'error': [],
            'x': [],
            'y': []
        },
        '0.5': {
            'error': [],
            'x': [],
            'y': []
        },
        '1.0': {
            'error': [],
            'x': [],
            'y': []
        }
    }

    directory = '../result_data/gp_only/'
    gp_only_files = os.listdir(directory)
    for filename in gp_only_files:
        mutation_rate = filename.split('_')[GP_MUTATION_RATE]
        crossover_rate = filename.split('_')[GP_CROSSOVER_RATE]
        if crossover_rate not in bar_for_crossover:
            continue
        if mutation_rate not in ['0.0', '0.1', '0.5']:
            continue

        with open(os.path.join(directory, filename)) as csv_file:
            reader = csv.DictReader(csv_file, skipinitialspace=True)
            test_error = [float(line['test error']) for line in reader]
            mean_test_error = sum(test_error) / len(test_error)
            var_test_error = np.var(test_error)
            bar_for_crossover[crossover_rate]['x'].append('Mutation rate: {}'.format(mutation_rate))
            bar_for_crossover[crossover_rate]['y'].append(mean_test_error)
            bar_for_crossover[crossover_rate]['error'].append(var_test_error)

    plotter = Plotter(title='GP crossover and mutation mean error', x_axis='Mutation rate', y_axis='Mean test error')
    for crossover_type in bar_for_crossover:
        data = bar_for_crossover[crossover_type]
        plotter.add_bar('Crossover: {}'.format(crossover_type), data['x'], data['y'], data['error'])
    plotter.plot()


if __name__ == '__main__':
    gp_only_bars_with_error()
