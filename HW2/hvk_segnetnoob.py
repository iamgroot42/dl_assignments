import numpy as np

from keras.layers import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from keras.layers import Dense, Flatten, Input, Dropout, Activation, Reshape, Permute
from keras.models import Sequential
from keras.optimizers import Adadelta


def addConv_BatchNorm2d(size, x, y, model, addInput = False):
	if (addInput):
		model.add(Convolution2D(size, x, y, border_mode = 'same', input_shape = (1, 256, 256)))
	else:
		model.add(Convolution2D(size, x, y, border_mode = 'same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))


def addConv_BatchNorm3d(size, x, y, z, model, addInput = False):
	if (addInput):
		model.add(Convolution3D(size, x, y, z, border_mode = 'same', input_shape = (1, 256, 256, 128)))
	else:
		model.add(Convolution3D(size, x, y, z, border_mode = 'same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))


def segnet2D_noob(n_labels=4):
	segnet2D_noob_model = Sequential()

	addConv_BatchNorm2d(64, 3, 3, segnet2D_noob_model, addInput = True)
	addConv_BatchNorm2d(64, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(MaxPooling2D())
	
	addConv_BatchNorm2d(128, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(128, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(MaxPooling2D())
	
	addConv_BatchNorm2d(256, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(256, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(256, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(MaxPooling2D())

	addConv_BatchNorm2d(512, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(512, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(512, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(MaxPooling2D())
	
	# Encoding done 
	encoder = segnet2D_noob_model

	segnet2D_noob_model.add(UpSampling2D())
	addConv_BatchNorm2d(512, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(512, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(256, 3, 3, segnet2D_noob_model)

	segnet2D_noob_model.add(UpSampling2D())
	addConv_BatchNorm2d(256, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(256, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(128, 3, 3, segnet2D_noob_model)

	segnet2D_noob_model.add(UpSampling2D())
	addConv_BatchNorm2d(128, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm2d(64, 3, 3, segnet2D_noob_model)

	segnet2D_noob_model.add(UpSampling2D())
	addConv_BatchNorm2d(64, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(Convolution2D(n_labels, 1, 1, border_mode='valid'))
	segnet2D_noob_model.add(BatchNormalization())
	
	segnet2D_noob_model.add(Reshape((n_labels, 256 * 256)))
	segnet2D_noob_model.add(Permute((2, 1)))
	segnet2D_noob_model.add(Activation('softmax'))

	segnet2D_noob_model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(lr = 0.01), metrics = ['accuracy','fmeasure'])
	return segnet2D_noob_model


def segnet3D_noob(n_labels=4):
	segnet3D_noob_model = Sequential()

	addConv_BatchNorm3d(64, 3, 3, 3, segnet3D_noob_model, addInput = True)
	addConv_BatchNorm3d(64, 3, 3, 3, segnet3D_noob_model)
	segnet3D_noob_model.add(MaxPooling3D())
	
	addConv_BatchNorm3d(128, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(128, 3, 3, 3, segnet3D_noob_model)
	segnet3D_noob_model.add(MaxPooling3D())
	
	addConv_BatchNorm3d(256, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(256, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(256, 3, 3, 3, segnet3D_noob_model)
	segnet3D_noob_model.add(MaxPooling3D())

	addConv_BatchNorm3d(512, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(512, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(512, 3, 3, 3, segnet3D_noob_model)
	segnet3D_noob_model.add(MaxPooling3D())
	
	# Encoding done 
	encoder = segnet3D_noob_model

	segnet3D_noob_model.add(UpSampling3D())
	addConv_BatchNorm3d(512, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(512, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(256, 3, 3, 3, segnet3D_noob_model)

	segnet3D_noob_model.add(UpSampling3D())
	addConv_BatchNorm3d(256, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(256, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(128, 3, 3, 3, segnet3D_noob_model)

	segnet3D_noob_model.add(UpSampling3D())
	addConv_BatchNorm3d(128, 3, 3, 3, segnet3D_noob_model)
	addConv_BatchNorm3d(64, 3, 3, 3, segnet3D_noob_model)

	segnet3D_noob_model.add(UpSampling3D())
	addConv_BatchNorm3d(64, 3, 3, 3, segnet3D_noob_model)
	segnet3D_noob_model.add(Convolution3D(n_labels, 1, 1, 1, border_mode='valid'))
	segnet3D_noob_model.add(BatchNormalization())
	

	segnet3D_noob_model.add(Reshape((n_labels, 256 * 256 * 128)))

	segnet3D_noob_model.add(Permute((2, 1)))
	segnet3D_noob_model.add(Activation('softmax'))

	segnet3D_noob_model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(lr = 0.01), metrics = ['accuracy','fmeasure'])
	return segnet3D_noob_model


if __name__ == '__main__':
	model = segnet3D_noob()
	print model.input
	print model.output
