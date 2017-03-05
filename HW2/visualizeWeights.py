import keras
from keras.models import load_model
from keras.utils.visualize_util import plot
from keras import backend as K
K.set_image_dim_ordering('th')

base = '/home/team1/DL/HW2/Models/'
models = ['s2/segment2.09-0.95.hdf5', 's3/segment3.09-0.91.hdf5', 'c2/classify2.09-0.72.hdf5', 'c3/classify3.09-0.91.hdf5']

for name in models:
	model = load_model(base + name)
	name = name[:-13]
	name = name[3:]
	plot(model, to_file=name + '.png')

