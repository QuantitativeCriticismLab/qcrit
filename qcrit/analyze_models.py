'''
Analyze models
'''
from functools import partial
from collections import OrderedDict
import os
import pickle
import csv

import numpy as np

from . import model_analyzer
from . import color as c

def _get_features(feature_data_file):
	#Obtain features that were previously mined and serialized into a file
	filename_to_features = None
	with open(feature_data_file, mode='rb') as pickle_file:
		filename_to_features = pickle.loads(pickle_file.read())
	return filename_to_features

def _get_file_classifications(classification_data_file):
	#Obtain classifications for each file
	with open(classification_data_file, mode='r') as classification_file:
		csv_reader = csv.reader(classification_file)
		filename_to_classification = {}
		label_val_to_label_name = OrderedDict(
			(tok.split(':')[1], tok.split(':')[0])
			for tok in next(csv_reader)
		)
		next(csv_reader)
		for line in csv_reader:
			filename_to_classification[line[0]] = line[1]
		assert all(v in label_val_to_label_name for v in filename_to_classification.values())
	return filename_to_classification, label_val_to_label_name

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

	if model_funcs is None: model_funcs = model_analyzer.DECORATED_ANALYZERS.keys()
	if not os.path.isfile(feature_data_file): raise ValueError('File "' + feature_data_file + '" does not exist')
	if not os.path.isfile(classification_data_file):
		raise ValueError('File "' + classification_data_file + '" does not exist')
	if not model_funcs: raise ValueError('No model analyzers were provided')
	if not all(f in model_analyzer.DECORATED_ANALYZERS for f in model_funcs):
		raise ValueError(
			'The values in set ' + str(set(model_funcs) - model_analyzer.DECORATED_ANALYZERS.keys()) +
			' are not among the decorated model analyzers in ' + str(model_analyzer.DECORATED_ANALYZERS.keys())
		)

	filename_to_features = _get_features(feature_data_file)

	filename_to_classification, label_val_to_label_name = _get_file_classifications(classification_data_file)

	if filename_to_features.keys() - filename_to_classification.keys():
		raise ValueError(
			'There exist some files in the feature_data_file for which no label exists in ' +
			'the classification_data_file: {\n\t' +
			'\n\t'.join(filename_to_features.keys() - filename_to_classification.keys()) + '\n}'
		)

	#Filter out unused labels (i.e. a label exists for a file with that name but no features were extracted for it)
	used_label_numbers = {filename_to_classification[filename] for filename in filename_to_features.keys()}
	label_val_to_label_name = OrderedDict(
		(k, v) for k, v in label_val_to_label_name.items() if k in used_label_numbers
	)

	#Convert features and classifications into sorted lists
	file_names = sorted([elem for elem in filename_to_features.keys()])
	feature_names = sorted(
		feature_name for feature_name in next(iter(filename_to_features.values())).keys()
	)

	data, target = _get_classifier_data(filename_to_features, filename_to_classification, file_names, feature_names)

	from timeit import timeit
	for funcname in model_funcs:
		print(
			'\n\n' + c.green(
				'Elapsed time: ' + '%.4f' % timeit(
					partial(
						model_analyzer.DECORATED_ANALYZERS[funcname], data, target, file_names,
						feature_names, label_val_to_label_name
					),
					number=1
				) + ' seconds'
			) + '\n'
		)
