import numpy as np


class Layer():
	def __init__(self):
		self.nodes = []

	def forward_pass(self):
		for node in self.nodes:
			node.forward_prop()

	def backward_pass(self):	
		for node in self.nodes:
			node.back_prop()