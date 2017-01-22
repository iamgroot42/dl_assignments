# from keras.datasets import mnist
import numpy as np
from Node import Sigmoid


if __name__ == "__main__":
	# (X_train, y_train), (X_test, y_test) = mnist.load_data()
	x = Sigmoid()
	y = Sigmoid()
	z = Sigmoid()
	x.add_node(z)
	y.add_node(z)
	x.input = 2.0
	y.input = 3.0
	x.forward_prop()
	y.forward_prop()
	z.gradient_in = 1.0
	z.back_prop()
	print z.input
	print x.gradient_in, y.gradient_in