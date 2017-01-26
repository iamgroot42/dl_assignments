import numpy as np
from helpers import sigmoid

class Layer():
	def __init__(self, weight_shape, activation):
		self.weights = np.random.rand(weight_shape[0], weight_shape[1]) / 100
		self.output = np.zeros(weight_shape)
		self.gradient = None
		self.dropout = 0.0
		self.momentum = 0.0
		self.activation = activation

	def forward(self, input_data):
		self.output = input_data * self.weights
		return self.activation.func(self.output)
