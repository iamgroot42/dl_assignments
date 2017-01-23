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

	def add_layer(self, layer):
		self.layers.append(layer)

	def forward_pass(self):
		self.layers[0].forward(self.X)
		for i in range(1,len(self.layers)):
			self.layers[i].forward(self.layers[i-1].output)
		self.Y_cap = self.layers[-1].output

	def backward_pass(self):
		self.layers[-1].gradient = self.error.gradient(self.Y, self.Y_cap)
		for i in range(len(self.layers)-2,0,-1):
			self.layers[i].gradient = self.layers[i].derivative(self.layers[i-1].output) * self.layers[i+1].gradient

	def update_weights(self):
		for layer in self.layers[:-1]:
			layer.momentum *= self.gamma
			layer.momentum += self.learning_rate * layer.gradient
			layer.weights -= layer.momentum

	def one_iter(self):
		self.forward_pass()
		self.backward_pass()
		self.update_weights()
		print self.error.error(self.Y, self.Y_cap)

	def train(self, X_train, Y_train):
		self.X = X_train
		self.Y = Y_train
		for i in range(self.iters):
			self.one_iter()

	def test(self):
		assert len(self.layers) >= 2
		self.layers[0].load_input_data(self.X_test)
		self.layers[-1].load_output_data(self.Y_test)
		self.forward_pass()
		accuracy = error.error(self.layers[-1])
		return accuracy
