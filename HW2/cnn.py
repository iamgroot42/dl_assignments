# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf
from tensorflow.python.platform import flags

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')


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
	model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
	return model


if __name__ == "__main__":
	m = volumeCNN()
