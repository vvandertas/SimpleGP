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

	def plot(self):
		layout = go.Layout(title=self.title,
						   xaxis=go.layout.XAxis(
							   title=go.layout.xaxis.Title(
								   text=self.x_axis
							   )
						   ),
						   yaxis=go.layout.YAxis(
							   title=go.layout.yaxis.Title(
								   text=self.y_axis
							   )
						   )
						   )
		py.plot({
			'data': self.traces,
			'layout': layout
		}, auto_open=True)


if __name__ == '__main__':
	plotter = Plotter()
	plotter.add_line('Line trace', [1, 2, 3, 4], [3, 3, 4, 5])
	plotter.add_scatter('Scatter trace ', [1, 2, 3], [5, 1, 2])
	plotter.plot()
