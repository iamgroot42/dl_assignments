import numpy as np


class Layer():
	def __init__(self, is_input_layer=False, is_output_layer=False):
		self.nodes = []
		self.is_input_layer = is_input_layer
		self.is_output_layer = is_output_layer

	def forward_pass(self):
		for node in self.nodes:
			node.forward_prop()

	def backward_pass(self):	
		for node in self.nodes:
			node.back_prop()

	def load_input_data(self, data):
		assert data.shape[1] == len(self.nodes)
		self.is_input_layer = True
		for i in range(len(self.nodes)):
			self.nodes[i].input = data[:,i]

	def load_output_data(self, data):
		assert data.shape[1] == len(self.nodes)
		self.is_output_layer = True
		for i in range(len(self.nodes)):
			self.nodes[i].output = data[:,i]

	def add_node(self, node):
		self.nodes.append(node)