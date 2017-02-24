import h5py

# Loading Train Data
base_dir = ''
train_file = h5py.File(base_dir + 'train_540k.mat')
train_data = train_file.get('training')

# Read One Index From the Data
index = 10
train_volume = train_data[:, index]

# Loading Labels
label_file_csf = h5py.File(base_dir + 'label_csf.mat')
label_data_csf = label_file_csf.get('label_csf')

label_file_wm = h5py.File(base_dir + 'label_wm.mat')
label_data_wm = label_file_wm.get('label_wm')

label_file_gm = h5py.File(base_dir + 'label_gm.mat')
label_data_gm = label_file_gm.get('label_gm')

# Reading One Index from Labels
def get_label(idx):
	if label_data_csf[0, idx] == 1:
		return 0
	elif label_data_wm[0, idx] == 1:
		return 1
	elif label_data_gm[0, idx] == 1:
		return 2

train_label = get_label(index)