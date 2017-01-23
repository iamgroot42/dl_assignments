import numpy as np

class Error():
	def __init__(self):
		self.error = 0.0

	def func_error(self, last_layer):
		for node in last_layer.nodes:
			self.error += np.sum(np.square(node.input-node.output))/2
		return error

	def derivative(self, last_layer):
		for node in last_layer.nodes:
			node.gradient_in = -(node.output - node.input)
