from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from Layer import Layer
from Activation import Sigmoid, SoftMax, ReLU
from Model import Model
from Error import Error

def process_data():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(60000, 784)
	y_train = y_train.reshape(60000, 1)
	X_test = X_test.reshape(10000, 784)
	y_test = y_test.reshape(10000, 1)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	X_test = np.matrix(X_test)
	y_test = np.matrix(y_test)
	return (X_train, y_train), (X_test, y_test)


def single_layer(X_train, X_test, y_train, y_test, verbose=False):
	m = Model(Error())
	m.add_layer(Layer((784,10), SoftMax(), dropout=0.2))
	t_acc = (1-m.train(X_train, y_train, verbose)) * 100
	print "Train accuracy", t_acc, "%"
	print "Test accuracy", (1-m.test(X_test, y_test)) * 100, "%"


def multi_layer(X_train, X_test, y_train, y_test, verbose=False):
	dropout_params = [None, 0.1, 0.2, 0.4]
	batch_sizes = [128, 256, 512]
	learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
	momentum_rates = [0.125, 0.25, 0.5]
	for dropout_param in dropout_params:
		for batch_size in batch_sizes:
			for learning_rate in learning_rates:
				for momentum_rate in momentum_rates:
					m = Model(Error(), learning_rate, momentum_rate, batch_size)
					m.add_layer(Layer((784,100), Sigmoid(), dropout=dropout_param))
					# m.add_layer(Layer((784,100), ReLU(), dropout=dropout_param))
					m.add_layer(Layer((100,10), SoftMax(), dropout=dropout_param))
					t_acc = (1-m.train(X_train, y_train, verbose)) * 100
					print "(", dropout_param, ",", batch_size, ",", learning_rate, ",", momentum_rate, ")"
					print "Train accuracy", t_acc, "%"
					print "Test accuracy", (1-m.test(X_test, y_test, 0.3)) * 100, "%"
					print "-------------"


if __name__ == "__main__":
	(X_train, y_train), (X_test, y_test) = process_data()
	multi_layer(X_train, X_test, y_train, y_test, False)
