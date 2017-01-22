import numpy as np
from Layer import Layer


class Model():
	def __init__(self, error, batch_size = 128, iters=5):
		self.layers = []
		self.error = error
		self.iters = iters
		self.batch_size = batch_size

	def load_data(self, X_train, Y_train, X_test, Y_test):
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test

	def add_layer(self, layer):
		self.layers.append(layer)

	def forward_pass(self):
		for layer in self.layers:
			layer.forward_pass()

	def backward_pass(self):
		for layer in self.layers:
			layer.backward_pass()

	def one_iter(self):
		self.forward_pass()
		self.backward_pass()
		for layer in self.layers[1:-1]:
			layer.clear()

	def train(self):
		assert len(self.layers) >= 2
		self.layers[0].load_input_data(self.X_train)
		self.layers[-1].load_output_data(self.Y_train)
		for i in range(self.iters):
			self.one_iter()

	def test(self):
		self.data = self.X_test
		self.outputs = self.Y_test
