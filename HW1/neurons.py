import numpy as np


class Model():
	def __init__(self, error, iters=100):
		self.layers = []
		self.error = error
		self.iters = iters

	def train(self, X_train, Y_train, X_test, Y_test):
		return

	def add_layer(self, layer):
		for neuron in self.layers:
			neuron.outgoing += layer
		for neuron in layer:
			neuron.incoming += self.layers


class Layer():
	def __init__(self, neuron, num_neurons = 0):
		self.neurons = []
		for _ in range(num_neurons):
			self.neurons.append(neuron())


class Node():
	def __init__(self):
		self.incoming = np.array([])
		self.outgoing = np.array([])
		self.output = np.zeros(())
		self.gradient = np.zeros(())
		self.dropout = False

	def func(self, x):
		return

	def gradient(self, x, wrt=None):
		return

	def forward_prop(self):
		output = np.zeros(self.func(incoming.output).shape)
		for incoming_node in self.incoming:
			output += self.func(incoming.output)

	def back_prop(self):
		gradient = np.zeros(self.gradient(outgoing.gradient,self).shape)
		for outgoing in self.outgoing:
			gradient += self.gradient(outgoing.gradient,self)

	def join_node(self, node):
		self.outgoing.append(node)
		node.incoming.append(self)


class Sigmoid(Node):
	def __init__(self):
		Node.__init__(self)

	def func(self, x):
		if self.dropout:
			return x * 0.0
		return 1.0 / ( 1 + np.exp(-x))

	def gradient(self, x, wrt=None):
		if self.dropout:
			return x * 0.0
		return self.gradient(x) * ( 1 - self.gradient(x))


class ReLU(Node):
	def __init__(self):
		Node.__init__(self)
		
	def func(self, x):
		if self.dropout:
			return x * 0.0
		return np.maximum(x, 0.0)

	def gradient(self, x, wrt=None):
		if self.dropout:
			return x * 0.0
		return 1.0 * (x > 0)


class Add(Node):
	def __init__(self):
		Node.__init__(self)
		
	def func(self, x):
		if self.dropout:
			return x * 0.0
		self.output = np.zeros(self.incoming.output.shape)
		for node in self.incoming:
			self.output += self.incoming.output
		return self.output

	def gradient(self, x, wrt=None):
		if self.dropout:
			return x*  0.0
		return np.ones(np.shape(x))


class Multiply(Node):
	def __init__(self):
		Node.__init__(self)
		
	def func(self, x):
		if self.dropout:
			return x * 0.0
		self.output = np.ones(self.incoming.output.shape)
		for node in self.incoming:
			self.output *= self.incoming.output
		return self.output

	def gradient(self, x, wrt=None):
		if self.dropout:
			return 0.0
		self.gradient = np.ones(self.incoming.output.shape, dtype=bool)
		# for i in self.incoming:
			
		# mask[5]=0
		# arr[mask]
		# for node in self.incoming:

		# return np.ones(np.shape(x))




# class MSE(Node):
# 	def __init__(self):
# 		Node.__init__(self)

# 	def func(self, x):
# 		return np.sum(np.square(true-predicted))/2