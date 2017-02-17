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

if __name__ == "__main__":
	print "main"
