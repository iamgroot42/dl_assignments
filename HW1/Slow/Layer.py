import numpy as np


class Layer():
	def __init__(self, is_input_layer=False, is_output_layer=False):
		self.nodes = []
		self.is_input_layer = is_input_layer
		self.is_output_layer = is_output_layer

	def create_layer(self, this_node, count):
		for _ in range(count):
			self.nodes.append(this_node())

	def clear(self):
		if self.is_input_layer:
			for node in self.nodes:
				node.gradient_in = 0.0
				node.output = None
		elif self.is_output_layer:
			for node in self.nodes:
				node.gradient_in = 0.0
				node.input = None
		else:
			for node in self.nodes:
				node.gradient_in = 0.0
				node.input = None
				node.output = None

	def forward_pass(self):
		i = 1
		for node in self.nodes:
			# print i
			node.forward_prop()
			i += 1

	def backward_pass(self):
		i = 1
		for node in self.nodes:
			# print i
			node.back_prop()
			i += 1

	def load_input_data(self, data):
		assert data.shape[1] == len(self.nodes)
		self.is_input_layer = True
		for i in range(len(self.nodes)):
			self.nodes[i].input = data[:,i]
			self.nodes[i].output = data[:,i]

	def load_output_data(self, data):
		assert data.shape[1] == len(self.nodes)
		self.is_output_layer = True
		for i in range(len(self.nodes)):
			self.nodes[i].output = data[:,i]

	def add_node(self, node):
		self.nodes.append(node)

	def update_weights(self, gamma, alpha):
		for node in self.nodes:
			if node.param:
				if node.momentum is None:
					node.momentum = 0.0
				node.momentum *= gamma
				node.momentum += alpha * node.gradient_in
				node.param -= node.momentum

	def join_layer(self, layer):
		for incoming in self.nodes:
			for outgoing in layer.nodes:
				incoming.add_node(outgoing)
