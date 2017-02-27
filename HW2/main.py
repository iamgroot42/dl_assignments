# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf
from tensorflow.python.platform import flags

import cnn
import read_data

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs',100, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 512, 'Batch size')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')


if __name__ == "__main__":
	xtr, ytr, xte, yte = read_data.load_data()
	# m = cnn.volumeCNN()
	# m.fit(xtr, ytr,
	# 	nb_epoch=FLAGS.nb_epochs,
	# 	batch_size=FLAGS.batch_size,
	# 	validation_split=0.2)
	# score = m.evaluate(xte, yte)[1]
	# print "\nTesting accuracy:",score*100,"%"
	m = cnn.plainAutoencoder(xtr, xte)
	m.fit(xtr, ytr,
		nb_epoch=FLAGS.nb_epochs,
		batch_size=FLAGS.batch_size,
		validation_split=0.2)
	score = m.evaluate(xte, yte)[1]
	print "\nTesting accuracy:",score*100,"%"
