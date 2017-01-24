from keras.datasets import mnist
import numpy as np
from Layer import Layer, Sigmoid
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
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	m = Model(Error())
	m.add_layer(Layer((784,100)))
	# m.add_layer(Sigmoid((100,100)))
	m.add_layer(Layer((100,10)))
	m.add_layer(Layer((10,1)))
	m.train(X_train, y_train)
