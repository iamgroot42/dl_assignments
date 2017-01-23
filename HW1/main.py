# from keras.datasets import mnist
import numpy as np
from Activation import Sigmoid, Node
from Linear import Linear
from Layer import Layer
from Model import Model
from Error import Error


if __name__ == "__main__":
	x = Node()
	y = Node()
	z = Node()
	q = Node()
	o = Node()
	f = Linear()
	x.add_node(q)
	y.add_node(q)
	q.add_node(f)
	z.add_node(f)
	f.add_node(o)
	l1 = Layer(is_input_layer=True)
	l2 = Layer()
	l3 = Layer()
	l4 = Layer(is_output_layer=True)
	l1.add_node(x)
	l1.add_node(y)
	l1.add_node(z)
	l2.add_node(q)
	l3.add_node(f)
	l4.add_node(o)
	data = np.array([[-2.0,5.0,-4.0]])
	labels = np.array([[-2.7]])
	m = Model(Error(), iters=5)
	m.add_layer(l1)
	m.add_layer(l2)
	m.add_layer(l3)
	m.add_layer(l4)
	m.load_data(data, labels, data, labels)
	m.train()
	# print f.func(f.input)
	# print f.func([1,2])
	# print f.gradient_in
