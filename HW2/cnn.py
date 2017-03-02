import keras

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.optimizers import Adadelta
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, Convolution3D, MaxPooling2D, MaxPooling3D, AveragePooling3D, UpSampling2D, UpSampling3D, ZeroPadding2D


def volumeCNN(learning_rate):
	model = Sequential()
	model.add(Convolution3D(16, 3, 3, 3, activation='relu', input_shape=(1, 13, 13, 13)))
	model.add(AveragePooling3D((2, 2, 2)))	
	model.add(Convolution3D(32, 3, 3, 3, activation='relu'))
	model.add(AveragePooling3D((2, 2, 2)))
	model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dropout(0.25))
	model.add(Dense(256))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.25))
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',
		optimizer=keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0),
		metrics=['accuracy'])
	return model


def paperCNN(learning_rate):
	model = Sequential()
	model.add(Convolution3D(64, 5, 5, 5, activation='relu', input_shape=(1, 13, 13, 13)))
	model.add(Convolution3D(256, 5, 5, 5, activation='relu'))
	model.add(Convolution3D(768, 5, 5, 5, activation='relu'))
	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(3))
	model.add(Activation('softmax'))
	model.add(Dropout(0.5))
	model.compile(loss='categorical_crossentropy',
		optimizer=keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0),
		metrics=['accuracy'])
	return model


if __name__ == "__main__":
	m = volumeCNN()
