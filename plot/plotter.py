import csv
import os

import plotly.graph_objs as go
import plotly.offline as py


class Plotter:
    def __init__(self, title='Plot', x_axis='X axis', y_axis='Y axis'):
        self.y_axis = y_axis
        self.x_axis = x_axis
        self.title = title
        self.traces = []

    def add_scatter(self, name, x, y):
        self.traces.append(go.Scatter(
            x=x, y=y, name=name, mode='markers'
        ))

    def add_line(self, name, x, y):
        self.traces.append(go.Scatter(
            x=x, y=y, name=name, mode='lines'
        ))

    def add_bar(self, name, x, y, error_values=None):
        if error_values:
            error_y = {
                'type': 'data',
                'array': error_values,
                'visible': True
            }
        else:
            error_y = None

        self.traces.append(go.Bar(
            name=name, x=x, y=y, error_y=error_y
        ))

    def plot(self):
        layout = go.Layout(title=self.title,
                           xaxis=go.layout.XAxis(
                               title=go.layout.xaxis.Title(
                                   text=self.x_axis
                               ),
                               type="category"
                           ),
                           yaxis=go.layout.YAxis(
                               title=go.layout.yaxis.Title(
                                   text=self.y_axis
                               )
                           ),
                           # Always group bars with the same x value
                           barmode='group'
                           )
        if not os.path.exists('../out'):
            os.mkdir('../out')
        py.plot({
            'data': self.traces,
            'layout': layout
        }, auto_open=True, filename='../out/temp-plot.html')

    def plot_csv(self, filename, x_col_name, y_col_name, trace_name='CSV plot', type=None):
        self.add_trace_from_csv(filename, x_col_name, y_col_name, trace_name, type)

        self.x_axis = x_col_name
        self.y_axis = y_col_name
        self.plot()

    def add_trace_from_csv(self, filename, x_col_name, y_col_name, trace_name=None, type='scatter'):
        if trace_name is None:
            trace_name = filename
        x = []
        y = []
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            for line in reader:
                x.append(line[x_col_name])
                y.append(line[y_col_name])
        if type == 'scatter':
            self.add_scatter(trace_name, x, y)
        else:
            self.add_line(trace_name, x, y)


if __name__ == '__main__':
    plotter = Plotter()
    plotter.plot_csv('../test.csv', 'mean nr of evaluations', 'average tree size')
