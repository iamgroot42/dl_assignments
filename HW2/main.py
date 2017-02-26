import cnn
import read_data

if __name__ == "__main__":
	m = cnn.volumeCNN()
	xtr,ytr,xte,yte = read_data.load_data()

