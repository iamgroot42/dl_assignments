import h5py
import numpy as np

from keras.utils.np_utils import to_categorical


def get_data(base_dir='Data/', splice=False):
	if not splice:
		return np.load(base_dir + "X.npy")
	else:
		return np.load(base_dir + "X_splice.npy")
	train_data = h5py.File(base_dir + 'train_540k.mat').get('training')[:]
	if splice:
		train_data = train_data.reshape(13, 13, 13*540000)
		train_data = np.transpose(train_data,(2,0,1))
		train_data = np.expand_dims(train_data, axis=1)
	else:
		train_data = train_data.reshape(13, 13, 13, 540000)
		train_data = np.transpose(train_data,(3,0,1,2))
		train_data = np.expand_dims(train_data, axis=1)
	return train_data


def split_data(X, y, split=0.8):
	train_examples = int(X.shape[0] * split)
	if len(X.shape) == 3:
		X_train = X[:train_examples,:]
		X_test = X[train_examples:,:]
		y_train = y[:train_examples,:]
		y_test = y[train_examples:,:]
	else:
		X_train = X[:train_examples,:]
		X_test = X[train_examples:,:]
		y_train = y[:train_examples,:]
		y_test = y[train_examples:,:]
	return X_train, y_train, X_test, y_test, get_weights(y_train)


def get_labels(base_dir='Data/', splice=False):
	if not splice:
		return np.load(base_dir + "y.npy")
	else:
		return np.load(base_dir + "y_splice.npy")
	label_csf = h5py.File(base_dir + 'label_csf.mat').get('label_csf')[0]
	label_wm = h5py.File(base_dir + 'label_wm.mat').get('label_wm')[0]
	label_gm = h5py.File(base_dir + 'label_gm.mat').get('label_gm')[0]

	one_hot = np.array([label_csf, label_wm, label_gm])
	one_hot = np.swapaxes(one_hot, 0, 1)
	if splice:
		one_hot = np.repeat(one_hot, 13, 0)
	return one_hot


def classfication_data(base_dir='Data/', split=0.8, splice=True):
	X = get_data(splice=splice)
	y = get_labels(splice=splice)
	return split_data(X, y, split)


def process_labels(y, n_classes=4):
	z = y.reshape(y.shape[0], np.product(y.shape[1:]))
	zee = []
	for x in z:
		zee.append(to_categorical(x, n_classes))
	zee = np.array(zee)
	zee = np.transpose(zee,(0,2,1))
	return zee


def get_weights(y):
	wd = {}
	labels = np.unique(y)
	weights = np.bincount(b.flatten().astype('int32')).astype('float64')
	weights = weights/weights.sum()
	weights = 1/weights
	weights = weights/weights.sum()
	for label in labels:
		wd[label] = weights[label]
	return wd


def segmentation_data(base_dir='IBSR_nifti_stripped/', splice=True):
	weights = {}
	if splice:
		xtr = np.load(base_dir + "X_train.npy")
		xte = np.load(base_dir + "X_test.npy")
		ytr = process_labels(np.load(base_dir + "y_train.npy"))
		yte = process_labels(np.load(base_dir + "y_test.npy"))
	else:
		xtr = np.load(base_dir + "X_train_splice.npy")
		xte = np.load(base_dir + "X_test_splice.npy")
		ytr = process_labels(np.load(base_dir + "y_train_splice.npy"))
		yte = process_labels(np.load(base_dir + "y_test_splice.npy"))
	return xtr, ytr, xte, yte, get_weights(ytr)
