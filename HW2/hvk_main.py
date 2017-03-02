# Tensorflow bug fix while importing keras
import tensorflow as tf
from PIL import Image
tf.python.control_flow_ops = tf
from tensorflow.python.platform import flags
from keras.callbacks import ModelCheckpoint

import cnn
import read_data
import hvk
import hvk_cnn

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


if __name__ == "__main__":
	print "Loading data"
	xtr, ytr, xte, yte = read_data.load_data()
	print "Data loaded"
	# m = hvk_cnn.volumeCNN()
	#m = cnn.volumeCNN(FLAGS.learning_rate)
	#m = cnn.paperCNN(FLAGS.learning_rate)
	m = hvk_cnn.paperCNN()
	# m = hvk.segnet2D_noob()
	# x = Image.fromarray(xtr[0])
	# x.saveImage("Original_image.png")
	m.fit(xtr, ytr,
		nb_epoch=FLAGS.nb_epochs,
		batch_size=FLAGS.batch_size,
		validation_split=0.2,
		callbacks=[ModelCheckpoint("papercnn.{epoch:02d}-{val_acc:.2f}.hdf5")])
	# res = m.predict(xtr)[0]
	# x = Image.fromarray(res)
	# x.saveImage("Segmented_image.png")
	score = m.evaluate(xte, yte)[1]
	print "\nTesting accuracy:",score*100,"%"
	#m = cnn.plainAutoencoder(xtr, xte)
	#m.fit(xtr, ytr,
	#	nb_epoch=FLAGS.nb_epochs,
	#	batch_size=FLAGS.batch_size,
	#	validation_split=0.2)
	#score = m.evaluate(xte, yte)[1]
	#print "\nTesting accuracy:",score*100,"%"
