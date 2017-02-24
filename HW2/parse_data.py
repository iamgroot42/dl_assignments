import sys
from keras.utils import io_utils


def parse_training_data(filename):
	return io_utils.HDF5Matrix(filename,'training')

def parse_labels(filename):
	id = '_'.join(filename.split('/')[-1].split('_')[:2])
	return io_utils.HDF5Matrix(filename, id)


if __name__ == "__main__":
	parse_labels(sys.argv[1])
	#parse_training_data(sys.argv[1])
