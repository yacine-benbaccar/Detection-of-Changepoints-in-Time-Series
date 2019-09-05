"""
author: @yacine-benbaccar
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from plotly.offline import plot
import plotly.graph_objects as go

class CPD():
	def __init__(self):
		self.trends = None
		self.intervals = None
		self.shapes = None
		self.max_depth = 5
		self.ratio = 0.01
		self.y = None
		self.dy = None
		self.regionColors = ["#baaeb5", "#fab505"]
		self.lineColors = []
		self.derivative = False

	def detect(self, y, ratio=0.01, max_depth=5, derivative=False,):
		"""
		y : np.1darray representing the values of the time series
		ratio: the minmum amout of points (in percentage) that composes an interval
		max_depth: determines the maximum number of intervals that we can discover (max_#interval = 2^max_depth)
		derivative: use the derivative of the signal to identify the changepoint intervals
		"""
		self.y = y.copy()
		self.derivative = derivative
		self.max_depth = max_depth
		self.ratio = ratio
		N = len(y)
		x = np.arange(N).reshape(-1,1)
		dy = np.gradient(y, 1)

		if derivative:
			reg = DecisionTreeRegressor(random_state=0, max_depth=max_depth, min_samples_leaf=int(N*ratio),)
			reg.fit(x,dy)
		else:
			reg = DecisionTreeRegressor(random_state=0, max_depth=max_depth, min_samples_leaf=int(N*ratio),)
			reg.fit(x,y)

		splits = np.insert(np.round(np.unique(reg.tree_.threshold)),
			[-1,0],
			[len(y),0])

		splits = np.sort(splits[splits>=0])
		ypred = np.array([])
		trends = []
		intervals = []
		shapes = []

		for i in range(len(splits)-1):
			min, max = int(splits[i]), int(splits[i+1])
			lr = LinearRegression()
			xlr, ylr = np.arange(min, max).reshape(-1,1), y[min:max]
			lr.fit(xlr, ylr)

			ypred = np.append(ypred, lr.predict(xlr))
			trends.append({
				"coefs": lr.coef_,
				"intercept": lr.intercept_
				})
			intervals.append([min,max])
			shapes.append(
				go.layout.Shape(
					type='rect',
					xref='x',
					yref='paper',
					x0=min,
					y0=0,
					x1=max,
					y1=1,
					layer='below',
					line=dict(
						dash='dash'
						),
					opacity=0.3,
					line_width=1,
					fillcolor=self.regionColors[i%2]))
		trends.append(ypred)

		self.shapes = shapes
		self.trends = trends
		self.intervals = intervals
		self.dy = dy
		return trends, intervals

	def plot(self, derivative=False, detrended=True,):
		N = len(self.y)
		fig = go.Figure()
		# coloring the regions of change 
		fig.update_layout(shapes=self.shapes)
		# plotting the real Time Series
		fig.add_trace(
			go.Scatter(
				x=np.arange(N),
				y=self.y,
				name="Real Time Series"))
		# plotting the detected Trends
		fig.add_trace(
			go.Scatter(
				x=np.arange(N),
				y=self.trends[-1],
				name="Detected Local Trends"))
		if detrended:
			fig.add_trace(
				go.Scatter(
					x=np.arange(N),
					y=self.y - self.trends[-1],
					name="Detrended Signal"))
		if derivative:
			fig.add_trace(
				go.Scatter(
					x=np.arange(N),
					y=self.dy,
					name="Derivative function"))
		fig.update_layout(
			title=go.layout.Title(
				text="Changepoints Detection, Derivative used = {}, ratio = {} %, max_depth = {}".format(self.derivative, self.ratio*100, self.max_depth),
				xref="paper",
				x=0.5))
		plot(fig)