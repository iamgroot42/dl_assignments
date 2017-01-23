import numpy as np


class Layer():
	def __init__(self, weight_shape=None):
		if weight_shape:
			self.weights = np.zeros(weight_shape)
			self.output = np.zeros(weight_shape)
		self.gradient = None
		self.dropout = 0.0
		self.momentum = 0.0

	def derivative(self, x):
		return self.weights

	def forward(self, input_data):
		print input_data.shape
		print self.weights.shape
		self.output = input_data * self.weights
		return self.output

	def random_init(self):
		self.weights = np.random.rand(self.weights.shape)


class Sigmoid(Layer):
	def __init__(self, weight_shape):
		Layer.__init__(self, weight_shape)

	def derivative(self, x):
		self.gradient = (1 / ( 1 + np.exp(-x))) * (1  - (1 / ( 1 + np.exp(-x))))
		return self.gradient

	def forward(self, input_data):
		self.output = 1 / ( 1 + np.exp(-input_data))
		return self.output


class ReLU(Layer):
	def __init__(self, weight_shape):
		Layer.__init__(self, weight_shape)
	
	def derivative(self, x):
		self.gradient = 1.0 * (x > 0)
		return self.gradient

	def func(self, input_data):
		self.output = np.maximum(input_data, 0.0)
		return self.output
