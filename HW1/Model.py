import numpy as np
from Layer import Layer


class Model():
	def __init__(self, error, learning_rate = 1e-1, gamma = 0.5, batch_size = 128, iters=5):
		self.layers = []
		self.error = error
		self.iters = iters
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.batch_size = batch_size

	def load_data(self, X_train, Y_train, X_test, Y_test):
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test
		assert len(self.layers) >= 2
		self.layers[0].load_input_data(self.X_train)
		self.layers[-1].load_output_data(self.Y_train)

	def add_layer(self, layer):
		self.layers.append(layer)

	def forward_pass(self):
		for layer in self.layers[:-1]:
			layer.forward_pass()

	def backward_pass(self):
		self.error.derivative(self.layers[-1])
		for layer in self.layers:
			layer.backward_pass()

	def update_weights(self):
		for layer in self.layers:
			layer.update_weights(self.gamma, self.learning_rate)

	def one_iter(self):
		self.forward_pass()
		self.backward_pass()
		self.update_weights()
		for layer in self.layers:
			layer.clear()

	def train(self):
		for i in range(self.iters):
			self.one_iter()

	def test(self):
		assert len(self.layers) >= 2
		self.layers[0].load_input_data(self.X_test)
		self.layers[-1].load_output_data(self.Y_test)
		self.forward_pass()
		accuracy = error.error(self.layers[-1])
		return accuracy
