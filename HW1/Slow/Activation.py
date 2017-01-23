import numpy as np
from Node import Node

class Activation(Node):
	def __init__(self):
		Node.__init__(self)


class Sigmoid(Activation):
	def __init__(self):
		Activation.__init__(self)

	def func(self, x):
		if self.dropout:
			return x * 0.0
		return 1.0 / ( 1 + np.exp(-x))

	def gradient(self, x, wrt=None):
		if self.dropout:
			return x * 0.0
		return self.func(x) * ( 1 - self.func(x))


class ReLU(Activation):
	def __init__(self):
		Activation.__init__(self)
		
	def func(self, x):
		if self.dropout:
			return x * 0.0
		return np.maximum(x, 0.0)

	def gradient(self, x, wrt=None):
		if self.dropout:
			return x * 0.0
		return 1.0 * (x > 0)
