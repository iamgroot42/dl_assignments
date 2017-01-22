# from keras.datasets import mnist
import numpy as np
from Node import Sigmoid
from Layer import Layer
from Model import Model
from Error import error


if __name__ == "__main__":
	# (X_train, y_train), (X_test, y_test) = mnist.load_data()
	x = Sigmoid()
	y = Sigmoid()
	z = Sigmoid()
	x.add_node(z)
	y.add_node(z)
	l1 = Layer()
	l2 = Layer()
	m = Model(error)
	l1.add_node(x)
	l1.add_node(y)
	l2.add_node(z)
	m.add_layer(l1)
	m.add_layer(l2)
	data = np.array([[2.0, 3.0],[3.0, 2.0]])
	labels = np.array([[0.0],[1.0]])
	m.load_data(data, labels, data, labels)
	m.train()
