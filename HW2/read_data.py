import h5py
import numpy as np


def get_data(base_dir='Data/'):
	train_data = h5py.File(base_dir + 'train_540k.mat').get('training')[0]
	train_data = train_data.reshape(13,13,13,540000)
	return train_data


def get_labels(base_dir='Data/'):
	label_csf = h5py.File(base_dir + 'label_csf.mat').get('label_csf')[0]
	label_wm = h5py.File(base_dir + 'label_wm.mat').get('label_wm')[0]
	label_gm = h5py.File(base_dir + 'label_gm.mat').get('label_gm')[0]

	one_hot = np.array([label_csf, label_wm, label_gm])
	one_hot = np.swapaxes(one_hot, 0, 1)
	return one_hot


if __name__ == "__main__":
	# X = get_data()
	y = get_labels()
