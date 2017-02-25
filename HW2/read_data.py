import h5py
import numpy as np


def get_data(base_dir='Data/', splice=False):
	train_data = h5py.File(base_dir + 'train_540k.mat').get('training')[0]
	if splice:
		train_data = train_data.reshape(13, 13, 13*540000)
	else:
		train_data = train_data.reshape(13, 13, 13, 540000)
	return train_data


def split_data(X, y, split=0.8):
	train_examples = X.shape[-1] * split
	if len(X.shape) == 3:
		X_train = X[:,:,:train_examples]
		X_test = X[:,:,train_examples:]
		y = np.repeat(y, 13, 0)
		y_train = y[:,:train_examples]
		y_test = y[:,train_examples:]
	else:
		X_train = X[:,:,:,:train_examples]
		X_test = X[:,:,:,train_examples:]
		y_train = y[:,:train_examples]
		y_test = y[:,train_examples:]
	return X_train, y_train, X_test, y_test


def get_labels(base_dir='Data/'):
	label_csf = h5py.File(base_dir + 'label_csf.mat').get('label_csf')[0]
	label_wm = h5py.File(base_dir + 'label_wm.mat').get('label_wm')[0]
	label_gm = h5py.File(base_dir + 'label_gm.mat').get('label_gm')[0]

	one_hot = np.array([label_csf, label_wm, label_gm])
	one_hot = np.swapaxes(one_hot, 0, 1)
	return one_hot


def load_data(base_dir='Data/', split=0.8):
	X = get_data()
	y = get_labels()
	return split_data(X, y, split)


if __name__ == "__main__":
	# X = get_data()
	y = get_labels()
