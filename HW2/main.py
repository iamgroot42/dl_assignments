# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from tensorflow.python.platform import flags
from keras.callbacks import ModelCheckpoint

import read_data
import classify_cnn
import segment_cnn

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('type', 'segment', 'Classification or segmentation')
flags.DEFINE_integer('dimension', 2, '2D or 3D')


def load_data(ttype, dimension):
	if ttype is 'classify':
		if dimension is 2:
			xtr, ytr, xte, yte = read_data.classfication_data(splice=True)
		else:
			xtr, ytr, xte, yte = read_data.classfication_data(splice=False)
	else:
		if dimension is 2:
			xtr, ytr, xte, yte = read_data.segmentation_data(splice=True)
		else:
			xtr, ytr, xte, yte = read_data.segmentation_data(splice=False)
	return xtr, ytr, xte, yte


def get_model(ttype, dimension):
	if ttype is 'classify':
		# Ideal learning rate: 1
		if dimension is 2:
			model = classify_cnn.CNN2D(FLAGS.learning_rate)
		else:
			model = classify_cnn.CNN3D(FLAGS.learning_rate)
	else:
		# Ideal learning rate: 0.01
		if dimension is 2:
			model = segment_cnn.CNN2D(FLAGS.learning_rate)
		else:
			model = segment_cnn.CNN3D(FLAGS.learning_rate)
	return model


if __name__ == "__main__":
	xtr, ytr, xte, yte = load_data(FLAGS.type, FLAGS.dimension)
	print ytr.shape
	model = get_model(FLAGS.type, FLAGS.dimension)
	print model.output
	exit()
	model.fit(xtr, ytr,
		nb_epoch=FLAGS.nb_epochs,
		batch_size=FLAGS.batch_size,
		validation_split=0.2,
		callbacks=[ModelCheckpoint("Models/" + FLAGS.type + str(FLAGS.dimension) + ".{epoch:02d}-{val_acc:.2f}.hdf5")])
	accuracy, f_score = model.evaluate(xte, yte)
	print "\nTesting accuracy:",accuracy*100,"%"
	print "\nF-Score accuracy:",f_score,"%"
