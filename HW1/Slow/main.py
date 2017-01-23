from keras.datasets import mnist
import numpy as np
from Activation import Sigmoid, Node
from Linear import Linear
from Layer import Layer
from Model import Model
from Error import Error


if __name__ == "__main__":
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(60000, 784)
	y_train = y_train.reshape(60000, 1)
	X_test = X_test.reshape(10000, 784)
	y_test = y_test.reshape(10000, 1)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	l1 = Layer(is_input_layer=True)
	l2 = Layer()
	l3 = Layer(is_output_layer=True)
	l1.create_layer(Node, 784)
	l2.create_layer(Node, 100)
	l3.create_layer(Node, 1)
	l1.join_layer(l2)
	l2.join_layer(l3)
	m = Model(Error(), iters=5)
	m.add_layer(l1)
	m.add_layer(l2)
	m.add_layer(l3)
	m.load_data(X_train, y_train, X_test, y_test)
	m.train()
	# print f.func(f.input)
	# print f.func([1,2])
	# print f.gradient_in
