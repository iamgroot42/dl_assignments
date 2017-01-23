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
		self.momentum = None

	def func(self, x):
		return x

	def gradient(self, x, wrt=None):
		return x

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
