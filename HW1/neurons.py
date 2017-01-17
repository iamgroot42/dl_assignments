# from keras.datasets import mnist
import numpy as np

# (X_train, y_train), (X_test, y_test) = mnist.load_data()


class Sigmoid():
	def __init__(self,x):
		self.x = x
		self.dropout = False

	def func(self):
		return 1.0 / ( 1 + np.exp(-self.x))

	def gradient(self):
		return self.gradient(self.x) * ( 1 - self.gradient(self.x))


class ReLU():
	def __init__(self,x):
		self.x = x
		self.dropout = False
		
	def func(self):
		return max(0.0, self.x)

	def gradient(self):
		if self.x > 0:
			return 1
		return 0


class Model():
	def __init__(self):
		self.layers = []

	def add_layer(self, x):
		self.layers.append(x)

	def train(self, x, y):
		