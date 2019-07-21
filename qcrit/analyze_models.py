'''
Analyze models
'''
from functools import partial
from collections import OrderedDict
import os
import pickle

import numpy as np

from .model_analyzer import DECORATED_ANALYZERS
from .color import GREEN, RESET

def _get_features(feature_data_file):
	#Obtain features that were previously mined and serialized into a file
	filename_to_features = None
	with open(feature_data_file, mode='rb') as pickle_file:
		filename_to_features = pickle.loads(pickle_file.read())
	return filename_to_features

#TODO handle backslash-commas in the csv they are meant to be read literally and do NOT
#demarcate a csv cell
def _get_file_classifications(classification_data_file):
	#Obtain classifications for each file
	filename_to_classification = {}
	with open(classification_data_file, mode='r') as classification_file:
		#TODO labels_key is a confusing variable name for a dictionary
		#TODO labels_key is declared in "with" block, but is returned external to it - fix this
		labels_key = OrderedDict(
			(np.float64(tok.split(':')[1]), tok.split(':')[0])
			for tok in classification_file.readline().strip().split(',')
		)
		classification_file.readline()
		for line in classification_file:
			line = line.strip().split(',')
			filename_to_classification[line[0]] = np.float64(line[1])
		assert all(v in labels_key.keys() for v in filename_to_classification.values())
	return filename_to_classification, labels_key

def _get_classifier_data(filename_to_features, filename_to_classification, file_names, feature_names):
	data_1d = [filename_to_features[file_name][feature] for file_name in file_names for feature in feature_names]
	data = []
	for i in range(len(file_names)):
		data.append([val for val in data_1d[i * len(feature_names): i * len(feature_names) + len(feature_names)]])
	target = [filename_to_classification[file_name] for file_name in file_names]

	assert data[-1][-1] == data_1d[-1]
	assert len(data) == len(target)
	assert len(data) == len(file_names)
	assert len(feature_names) == len(data[0])

	#Convert lists to numpy arrays so they can be used in the machine learning models
	data = np.asarray(data)
	target = np.asarray(target)
	return (data, target)

#TODO unit test this
def main(feature_data_file, classification_data_file, model_funcs=None):
	'''Runs all decorated model analyzers'''

	if model_funcs is None: model_funcs = DECORATED_ANALYZERS.keys()
	if not os.path.isfile(feature_data_file): raise ValueError('File "' + feature_data_file + '" does not exist')
	if not os.path.isfile(classification_data_file):
		raise ValueError('File "' + classification_data_file + '" does not exist')
	if not model_funcs: raise ValueError('No model analyzers were provided')
	if not all(f in DECORATED_ANALYZERS for f in model_funcs):
		raise ValueError(
			'The values in set ' + str(set(model_funcs) - DECORATED_ANALYZERS.keys()) +
			' are not among the decorated model analyzers in ' + str(DECORATED_ANALYZERS.keys())
		)

	filename_to_features = _get_features(feature_data_file)

	filename_to_classification, labels_key = _get_file_classifications(classification_data_file)

	if filename_to_features.keys() - filename_to_classification.keys():
		raise ValueError(
			'There exist some files for which no label exists: {\n\t' +
			'\n\t'.join(filename_to_features.keys() - filename_to_classification.keys()) + '\n}'
		)

	#Filter out unused labels (i.e. a label exists but no files are assigned that label)
	#TODO we probably don't want to filter here, instead we could remove these two lines,
	#	and fix the divide by zero issue
	#TODO as a special case in ml_analyzers
	used_label_numbers = {filename_to_classification[filename] for filename in filename_to_features.keys()}
	labels_key = OrderedDict((k, v) for k, v in labels_key.items() if k in used_label_numbers)

	#Convert features and classifications into sorted lists
	file_names = sorted([elem for elem in filename_to_features.keys()])
	#TODO we're doing repeated work
	feature_names = sorted(list({
		feature for feature_to_val in filename_to_features.values()
		for feature in feature_to_val.keys()
	}))

	data, target = _get_classifier_data(filename_to_features, filename_to_classification, file_names, feature_names)

	from timeit import timeit
	for funcname in model_funcs:
		print(
			'\n\n' + GREEN + 'Elapsed time: ' + '%.4f' %
			timeit(
				partial(DECORATED_ANALYZERS[funcname], data, target, file_names, feature_names, labels_key), number=1
			)
			+ ' seconds' + RESET + '\n'
		)
