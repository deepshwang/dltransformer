import pickle
import gzip
import os
import yaml

def save_pickle(filename, data):
	directory = os.path.dirname(filename)
	if not os.path.exists(directory):
		os.makedirs(directory)

	with gzip.open(filename, 'wb') as f:
		pickle.dump(data, f)


def load_pickle(filename):
	with gzip.open(filename, 'rb') as f:
		data = pickle.load(f)

	return data

def open_yaml(filename):
	with open(filename) as f:
		out = yaml.load(f, Loader=yaml.FullLoader)
	return out
