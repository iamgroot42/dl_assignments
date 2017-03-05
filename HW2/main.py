# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from tensorflow.python.platform import flags
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score

import read_data
import classify_cnn
import segment_cnn
import segmentweak_cnn

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('type', 'segment', 'Classification or segmentation')
flags.DEFINE_integer('dimension', 2, '2D or 3D')
flags.DEFINE_string('model_path', "", 'Path to saved model with weights')

def load_data(ttype, dimension):
	if ttype == "classify":
		if dimension is 2:
			xtr, ytr, xte, yte, weights = read_data.segmentation_data(splice=True)
			ytr_slice, yte_slice = [], []
			for i in xrange(ytr.shape[0]):
				temp = np.sum(ytr[i], axis = 0)
				ytr_slice.append(temp)
			for i in xrange(yte.shape[0]):
				temp = np.sum(yte[i], axis = 0)
				yte_slice.append(temp)
			ytr, yte = np.array(ytr_slice), np.array(yte_slice)
			ytr /= ytr.sum()
			yte /= yte.sum() 
			ytr = np.multiply(ytr, weights.values())
			yte = np.multiply(yte, weights.values())
			for i in xrange(ytr.shape[0]):
				pos = np.argmax(ytr[i])
				ytr[i] = [0 if (pos != j) else 1 for j in xrange(len(ytr[i]))]
			for i in xrange(yte.shape[0]):
				pos = np.argmax(yte[i])
				yte[i] = [0 if (pos != j) else 1 for j in xrange(len(yte[i]))]
		else:
			xtr, ytr, xte, yte, weights = read_data.classfication_data(splice=False)
	else:
		if dimension is 2:
			xtr, ytr, xte, yte, weights = read_data.segmentation_data(splice=True, type=3)
		else:
			xtr, ytr, xte, yte, weights = read_data.segmentation_data(splice=False, type=3)
	return xtr, ytr, xte, yte, weights


def get_model(ttype, dimension):
	if ttype == "classify":
		# Ideal learning rate: 1
		if dimension is 2:
			model = classify_cnn.CNN2D(learning_rate=FLAGS.learning_rate)
		else:
			model = classify_cnn.CNN3D(learning_rate=FLAGS.learning_rate)
	else:
		# Ideal learning rate: 0.01
		if dimension is 2:
			model = segmentweak_cnn.CNN2D(learning_rate = FLAGS.learning_rate)
		else:
			model = segmentweak_cnn.CNN3D(learning_rate = FLAGS.learning_rate)
	return model


def test_model(model, X, y):
	accuracy, f_score = model.evaluate(X,y,batch_size=FLAGS.batch_size)[1:]
	print "\nTesting accuracy:",accuracy*100,"%"
	print "F-Score accuracy:",f_score


def classwise_scores(model, X, y):
	y_ =  model.predict(X, batch_size = FLAGS.batch_size)
	acc = {}
	if FLAGS.type == "classify":
		for i in range(y.shape[1]):
			acc[i] = (1 * (y[:,i] == (1 * (y_[:,i] == np.amax(y_,1))))).sum()
			acc[i] /= float(np.prod(y.shape)/y.shape[1])
	else:	
		for i in range(y.shape[2]):
			acc[i] = (1 * (y[:,:,i].flatten() == (1*(y_[:,:,i] == np.amax(y_,2))).flatten())).sum()
			acc[i] /= float(np.prod(y.shape)/y.shape[2])
	return acc


if __name__ == "__main__":
	xtr, ytr, xte, yte, weights = load_data(FLAGS.type, FLAGS.dimension)
	if FLAGS.model_path == "":
		model = get_model(FLAGS.type, FLAGS.dimension)
		if FLAGS.type == "classify":
			model.fit(xtr, ytr,
				nb_epoch=FLAGS.nb_epochs,
				batch_size=FLAGS.batch_size,
				validation_split=0.2,
				#,class_weight=weights
				callbacks=[ModelCheckpoint("Models/" + FLAGS.type + str(FLAGS.dimension) + ".{epoch:02d}-{val_acc:.2f}.hdf5")]
				)
		else:
			model.fit(xtr, ytr,
				nb_epoch=FLAGS.nb_epochs,
				batch_size=FLAGS.batch_size,
				validation_split=0.2
				#,class_weight=weights
				,callbacks=[ModelCheckpoint("Models/" + FLAGS.type + str(FLAGS.dimension) + ".{epoch:02d}-{val_acc:.2f}.hdf5")]
				)
	else:
		model = load_model(FLAGS.model_path)
	test_model(model, xte, yte)
	acc = classwise_scores(model, xte, yte)
	print "Classwise accuracies:"
	print acc
