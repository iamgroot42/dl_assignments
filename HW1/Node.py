import numpy as np


class Node():
	def __init__(self):
		self.gradient_in = 0.0
		self.input = None
		self.output = None
		self.incoming = []
		self.outgoing = []
		self.dropout = False
		self.param = None

	def func(self, x):
		return x

	def gradient(self, x, wrt=None):
		return x * 0.0

	def forward_prop(self):
		self.output = self.func(self.input)
		for outgoing_node in self.outgoing:
			if outgoing_node.input is None:
				outgoing_node.input = np.zeros(self.output.shape)
			outgoing_node.input += self.output

	def back_prop(self):
		for incoming_node in self.incoming:
			incoming_node.gradient_in += self.gradient(incoming_node.output, incoming_node) * self.gradient_in

	def add_node(self, node):
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
		return self.func(x) * ( 1 - self.func(x))


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


class Multiply(Node):
	def __init__(self):
		Node.__init__(self)
		
	def func(self, x):
		if self.dropout:
			return  0.0
		output = np.sum(self.incoming.output) * self.param

	def gradient(self, x, wrt=None):
		if self.dropout:
			return 0.0
		return self.param


class Add(Node):
	def __init__(self):
		Node.__init__(self)
		
	def func(self, x):
		if self.dropout:
			return x * 0.0
		output = np.zeros(self.incoming.output.shape)
		for node in self.incoming:
			output += self.incoming.output
		return output

	def gradient(self, x, wrt=None):
		if self.dropout:
			return x * 0.0
		return np.ones(np.shape(x))
