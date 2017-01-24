import numpy as np
from helpers import sigmoid

class Layer():
	def __init__(self, weight_shape):
		self.weights = np.random.rand(weight_shape[0], weight_shape[1])
		self.output = np.zeros(weight_shape)
		self.gradient = None
		self.dropout = 0.0
		self.momentum = 0.0

	def activation(self, x):
		return x

	def activation_grad(self, x):
		return 1.0

	def backward(self, x):
		self.gradient = np.transpose(self.weights)
		return self.gradient

	def forward(self, input_data):
		self.output = input_data * self.weights
		return self.output


class Sigmoid(Layer):
	def __init__(self, weight_shape):
		Layer.__init__(self, weight_shape)

	def activation(self, x):
		self.output = sigmoid(x)
		return self.output

	def activation_grad(self, x):
		self.gradient = np.multiply(sigmoid(x), 1 - sigmoid(x))
		return self.gradient


class ReLU(Layer):
	def __init__(self, weight_shape):
		Layer.__init__(self, weight_shape)

	def activation(self, x):
		self.output = np.maximum(x, 0.0)
		return self.output

	def activation_grad(self, x):
		self.gradient = 1.0 * (x > 0)
		return self.gradient

