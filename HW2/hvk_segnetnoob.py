mport numpy as np

from keras.layers import BatchNormalization
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers import Dense, Flatten, Input, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adadelta


def addConv_BatchNorm(size, x, y, model, addInput = False):
	if (addInput):
		model.add(Convolution2D(size, x, y, border_mode = 'same', input_shape = (1, size, size)))
	else:
		model.add(Convolution2D(size, x, y, border_mode = 'same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))


# def addConv_BatchNorm(size, x, y, model):
# 	model.add(Deconvolution2D(size, x, y, input_shape = model.layers[-1].output_shape, border_mode = 'same', output_shape = model.layers[-1].output_shape))
# 	model.add(BatchNormalization())
# 	model.add(Activation('relu'))


def segnet2D_noob():
	padding = {'top_pad' : 2, 'bottom_pad' : 1, 'left_pad' : 2, 'right_pad' : 1}
	segnet2D_noob_model = Sequential()

	addConv_BatchNorm(13, 3, 3, segnet2D_noob_model, addInput = True)
	addConv_BatchNorm(13, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(MaxPooling2D(border_mode = 'same'))

	addConv_BatchNorm(6, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm(6, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(MaxPooling2D(border_mode = 'same'))

	addConv_BatchNorm(3, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm(3, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm(3, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(MaxPooling2D(border_mode = 'same'))

	#addConv_BatchNorm(2, 3, 3, segnet2D_noob_model)
	#addConv_BatchNorm(2, 3, 3, segnet2D_noob_model)
	#addConv_BatchNorm(2, 3, 3, segnet2D_noob_model)
	#segnet2D_noob_model.add(MaxPooling2D(border_mode = 'same'))

	# Encoding done 
	encoder = segnet2D_noob_model

	# Check if deconv required or conv only
	addConv_BatchNorm(3, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm(3, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm(3, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(UpSampling2D())

	addConv_BatchNorm(6, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm(6, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm(6, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(UpSampling2D())

	addConv_BatchNorm(13, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm(13, 3, 3, segnet2D_noob_model)
	addConv_BatchNorm(13, 3, 3, segnet2D_noob_model)
	segnet2D_noob_model.add(UpSampling2D())

	# addConv_BatchNorm(16, 3, 3, segnet2D_noob_model)
	# addConv_BatchNorm(16, 3, 3, segnet2D_noob_model)
	# addConv_BatchNorm(16, 3, 3, segnet2D_noob_model)
	# segnet2D_noob_model.add(UpSampling2D())

	segnet2D_noob_model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-08, decay = 0.0), metrics = ['accuracy'])
	return segnet2D_noob_model


if __name__ == '__main__':
	model = segnet2D_noob()
