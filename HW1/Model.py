import numpy as np


class Model():
	def __init__(self, error, batch_size = 128, iters=100):
		self.layers = []
		self.error = error
		self.iters = iters
		self.batch_size = batch_size
		self.input = []

	def load_data(self, X_train, Y_train, X_test, Y_test):
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test

	def add_layer(self, layer):
		self.layer.append(layer)

	def train(self):
		self.data = self.X_train
		self.outputs = self.Y_train

	def test(self):
		self.data = self.X_test
		self.outputs = self.Y_test
