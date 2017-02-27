import keras

from keras.optimizers import Adadelta
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, Convolution3D, MaxPooling2D, UpSampling2D, ZeroPadding2D


def plainAutoencoder(X_train, X_test, learning_rate=1.0):
	padding = {
		'top_pad':2,
		'bottom_pad':1,
		'left_pad':2,
		'right_pad':1
	}
	input_img = Input(shape=(13, 13, 13))
	x = ZeroPadding2D(padding=padding)(input_img)
	print x.get_shape()
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
	encoder = Model(input=input_img, output=encoded)
	autoencoder = Model(input=input_img, output=decoded)
	# Configure autoencoder
	autoencoder.compile(loss='binary_crossentropy',optimizer=Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
, metrics=['accuracy'])
	autoencoder.fit(X_train, X_train,
				nb_epoch=100,
				batch_size=256,
				validation_split=0.2)
	score = autoencoder.evaluate(X_test, X_test)[1]
	print("\nAutoencoder accuracy: " + str(score))
	# Build ultimate model
	for i in encoder.layers:
		i.trainable = False
	final_model = Sequential()
	final_model.add(encoder)
	final_model.add(Flatten())
	final_model.add(Dense(256))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.2))
	final_model.add(Dense(64))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.2))
	final_model.add(Dense(3))
	final_model.add(Activation('softmax'))
	return model


def volumeAutoencoder(X_train, X_test):
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
	model.add(Activation('sigmoid'))
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
