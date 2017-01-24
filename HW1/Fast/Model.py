import numpy as np
from Layer import Layer


class Model():
	def __init__(self, error, learning_rate = 1e-3, gamma = 0.5, batch_size = 128, iters=1):
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
		# print self.layers[0].output.shape
		for i in range(1,len(self.layers)):
			self.layers[i].forward(self.layers[i-1].output)
			# print self.layers[i].output.shape
		self.Y_cap = self.layers[-1].output

	def backward_pass(self):
		self.layers[-1].gradient = self.layers[-1].derivative(self.layers[-2].output) * self.error.gradient(self.Y, self.Y_cap)
		for i in range(len(self.layers)-2,0,-1):
			self.layers[i].gradient = np.multiply(np.transpose(self.layers[i].derivative(self.layers[i-1].output)), self.layers[i+1].gradient)
		self.layers[0].gradient = self.layers[1].gradient * self.layers[0].derivative(self.X)
		for layer in self.layers:
			print layer.gradient.shape

	def update_weights(self):
		for layer in self.layers:
			layer.momentum *= self.gamma
			print "wololo"
			layer.momentum += self.learning_rate * layer.gradient
			print layer.momentum.shape
			print layer.weights.shape
			print "wololo"
			layer.weights -= layer.momentum
			print "wololo"

	def one_iter(self):
		self.forward_pass()
		self.backward_pass()
		self.update_weights()
		print self.error.error(self.Y, self.Y_cap)

	def train(self, X_train, Y_train):
		self.X = X_train[0]
		self.Y = Y_train[0]
		for i in range(self.iters):
			self.one_iter()

	def test(self):
		self.X = X_train
		self.Y = Y_train
		self.one_iter()
		accuracy = self.error.error(self.Y, self.Y_cap)
		return accuracy
