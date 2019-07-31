'''
Utilities for a featuer extraction
'''

import pickle
import os
from os.path import join
from io import StringIO

from tqdm import tqdm

from .color import GREEN, YELLOW, RESET
from .textual_feature import decorated_features, clear_cache, tokenize_types, debug_output

def parse_tess(file_name):
	'''Used to parse tess tags found at the beginning of lines of .tess files'''
	file_text = StringIO()
	with open(file_name, mode='r', encoding='utf-8') as file:
		for line in file:
			#Ignore lines without tess tags, or parse the tag out and strip whitespace
			if not line.startswith('<'):
				continue
			assert '>' in line
			file_text.write(line[line.index('>') + 1:].strip())
			file_text.write(' ')
	return file_text.getvalue()

FILE_PARSERS = {
	'tess': parse_tess,
}

def _get_filenames(corpus_dir, file_extension, excluded_paths):
	#Obtain all the files to parse by traversing through the corpus directory
	file_names = []
	for current_path, current_dir_names, current_file_names in os.walk(corpus_dir, topdown=True):
		#Remove the excluded directories to prevent traversing into them
		del_indexes = []
		for i, cur_dir in enumerate(current_dir_names):
			if join(current_path, cur_dir) + os.sep in excluded_paths:
				del_indexes.append(i)
		#Iterate backwards to prevent index removal issues
		for i in range(len(del_indexes) - 1, -1, -1):
			del current_dir_names[del_indexes[i]]

		for current_file_name in current_file_names:
			if current_file_name.endswith('.' + file_extension) \
			and join(current_path, current_file_name) not in excluded_paths:
				file_names.append(join(current_path, current_file_name))
	return sorted(file_names)

def _extract_features(corpus_dir, file_extension, excluded_paths, features, output_file):
	file_names = _get_filenames(corpus_dir, file_extension, excluded_paths)
	feature_tuples = [(name, decorated_features[name]) for name in features]
	text_to_features = {} #Associates file names to their respective features
	print('Extracting features from .' + file_extension + ' files in ' + YELLOW + corpus_dir + RESET)

	#Feature extraction
	for file_name in file_names if output_file is None else tqdm(file_names, dynamic_ncols=True):
		text_to_features[file_name] = {}
		file_text = FILE_PARSERS[file_extension](file_name)

		for feature_name, feature_func in feature_tuples:
			score = feature_func(text=file_text, filepath=file_name)
			text_to_features[file_name][feature_name] = score
			if output_file is None:
				print(file_name + ', ' + str(feature_name) + ', ' + GREEN + str(score) + RESET)

	clear_cache(tokenize_types, debug_output)

	if output_file is not None:
		print(
			'Feature mining complete. Attempting to write feature results to "' +
			YELLOW + output_file + RESET + '"...'
		)
		with open(output_file, 'wb') as pickle_file:
			pickle_file.write(pickle.dumps(text_to_features))
		print(GREEN + 'Success!' + RESET)

# file_extension must not include the dot ('.')
# If excluded_paths is given, it must be a set and it can contain files or directories (the directories must
# end in a file separator e.g. slash on Mac or Linux)
def main(corpus_dir, file_extension, excluded_paths=None, features=None, output_file=None):
	'''Run feature extraction on all decorated features'''
	if excluded_paths is None: excluded_paths = set()
	if features is None: features = decorated_features.keys()

	if not corpus_dir: raise ValueError('Must provide a directory that contains the corpus')
	if not file_extension: raise ValueError('Must provide a file extension of files to parse')
	if not features: raise ValueError('No features were provided')
	if not os.path.isdir(corpus_dir): raise ValueError('Path "' + corpus_dir + '" is not a valid directory')
	if not file_extension in FILE_PARSERS:
		raise ValueError('"' + str(file_extension) + '" is not an available file extension to parse')
	if not isinstance(excluded_paths, set): raise ValueError('Excluded paths must be in a set')
	if not all(os.path.isfile(path) or os.path.isdir(path) for path in excluded_paths):
		raise ValueError('Each path in ' + str(excluded_paths) + ' must be a valid path for a file or directory!')
	if not all(name in decorated_features.keys() for name in features):
		raise ValueError(
			'The values in set ' + str(set(features) - decorated_features.keys()) +
			' are not among the decorated features in ' + str(decorated_features.keys())
		)

	if output_file:
		if not isinstance(output_file, str): raise ValueError('Output file must be a string for a file path')
		if os.path.isfile(output_file): raise ValueError('Output file "' + output_file + '" already exists!')
		if os.path.isdir(output_file):
			raise ValueError('The end of the path "' + output_file + '" is a directory - please specify a filename')
		if os.sep in output_file and not os.path.isdir(os.path.dirname(output_file)):
			raise ValueError('"' + os.path.dirname(output_file) + '" is not a valid directory!')
	elif output_file is not None: raise ValueError('Output file must be truthy, or None')

	from timeit import timeit
	from functools import partial
	print(
		'\n\n' + GREEN + 'Feature mining elapsed time: ' + '%.4f' % timeit(
			partial(_extract_features, corpus_dir, file_extension, excluded_paths, features, output_file),
			number=1
		) +
		' seconds' + RESET
	)
