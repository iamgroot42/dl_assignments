import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D


def plainAutoencoder():
	model = Sequential()
	return model


def volumeAutoencoder():
	model = Sequential()
	return model


def plainCNN():
	model = Sequential()
	# model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(3, 32, 32)))
	return model


def volumeCNN():
	model = Sequential()
	model.add(Convolution2D(16, 2, 2, activation='relu', input_shape=(13, 13, 13)))
	model.add(Convolution2D(32, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dropout(0.25))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',
		optimizer=keras.optimizers.Adadelta(lr=0.01, rho=0.95, epsilon=1e-08, decay=0.0),
		metrics=['accuracy'])
	return model


if __name__ == "__main__":
	m = volumeCNN()
