import csv
import os
import re

from plot.plotter import Plotter


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


if __name__ == '__main__':
    gp_only()
    plot_global_results()
