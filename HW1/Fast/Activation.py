import numpy as np

class Linear():
	def func(self, x):
		return x

	def gradient(self, x):
		return (x * 0.0) + 1.0


class Sigmoid():
	def func(self, x):
		return 1.0 / ( 1 + np.exp(-x))

	def gradient(self, x):
		return self.func(x) * ( 1 - self.func(x))


class ReLU():
	def func(self, x):
		return np.maximum(x, 0.0)

	def gradient(self, x):
		return 1.0 * (x > 0)
