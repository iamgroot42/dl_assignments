import numpy as np 
from Node import Node


class Linear(Node):
	def __init__(self):
		Node.__init__(self)
		self.param = 2.0 * np.ones(())
	
	def forward_prop(self):
		self.output = self.func([node.output for node in self.incoming])
		for outgoing_node in self.outgoing:
			if outgoing_node.input is None:
				outgoing_node.input = np.zeros(self.output.shape)
			outgoing_node.input += self.output

	def func(self, x):
		if self.dropout:
			return  x * 0.0
		output = np.sum([ node for node in x]) * self.param
		return output

	def gradient(self, x, wrt=None):
		if self.dropout:
			return 0.0
		return self.param
