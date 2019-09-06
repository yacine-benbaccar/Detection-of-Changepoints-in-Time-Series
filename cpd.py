"""
author: @yacine-benbaccar
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from scipy.signal import savgol_filter
from plotly.offline import plot
import plotly.graph_objects as go

class CPD():
	def __init__(self):
		self.trends = None
		self.intervals = None
		self.shapes = None
		self.max_depth = 5
		self.ratio = 0.01
		self.criterion = "mse"
		self.y = None
		self.dy = None
		self.regionColors = ["#baaeb5", "#fab505"]
		self.lineColors = []
		self.derivative = False
		self.linear_model = 'Linear'

	def detect(self, y, ratio=0.01, max_depth=5, derivative=False, criterion="mse",linear_model="Linear",**kwargs):
		"""
		y : np.1darray representing the values of the time series
		ratio: the minmum amout of points (in percentage) that composes an interval
		max_depth: determines the maximum number of intervals that we can discover (max_#interval = 2^max_depth)
		derivative: use the derivative of the signal to identify the changepoint intervals
		criterion: the loss metric to quantify the error of our splitting intervals ["mae", "mse"], this is used for the detection of intervals
		linear_model: ['Linear', 'Ridge', 'Lasso', 'ElasticNet'], this is used for the prediction of local trends, all models are used with their 
			default parameters, there is no need for more elaborate models as we do not want to make prediction just to find the overall trend of
			the signal
		"""
		self.y = y.copy()
		self.derivative = derivative
		self.max_depth = max_depth
		self.ratio = ratio
		self.criterion = criterion
		self.linear_model = linear_model
		N = len(y)
		x = np.arange(N).reshape(-1,1)
		dy = np.gradient(y, 1)

		if derivative:
			reg = DecisionTreeRegressor(random_state=0, max_depth=max_depth, min_samples_leaf=int(N*ratio), criterion=criterion,)
			reg.fit(x,dy)
		else:
			reg = DecisionTreeRegressor(random_state=0, max_depth=max_depth, min_samples_leaf=int(N*ratio), criterion=criterion,)
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

			if linear_model == "Linear":
				lr = LinearRegression()
			elif linear_model == "Lasso":
				lr = Lasso()
			elif linear_model == "Ridge":
				lr = Ridge()
			elif linear_model == "ElasticNet":
				lr = ElasticNet()

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

	def plot(self, derivative=False, detrended=False, smooth=False,):
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
		if smooth:
			fig.add_trace(
				go.Scatter(
					x=np.arange(N),
					y=self.smooth_trend(),
					name="Smoothed Detected Local Trends"))
		fig.update_layout(
			title=go.layout.Title(
				text="Changepoints Detection, Derivative used = {}, ratio = {} %, max_depth = {}, DT_criterion = {}, Linear_model = {}".format(self.derivative, self.ratio*100, self.max_depth, self.criterion, self.linear_model),
				xref="paper",
				x=0.5))
		plot(fig)

	def smooth_trend(self):
		return savgol_filter(self.trends[-1], int(len(self.y)*0.01), polyorder=5)