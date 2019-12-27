'''
Utilities for a feature extraction
'''

import pickle
import os
from os.path import join
from io import StringIO
import collections as clctn

from tqdm import tqdm

from . import color as c
from . import textual_feature

def parse_tess(file_name):
	'''Used to parse tess tags found at the beginning of lines of .tess files'''
	file_text = StringIO()
	with open(file_name, mode='r', encoding='utf-8') as file:
		for line in file:
			#Ignore lines without tess tags, or parse the tag out and strip whitespace
			if not line.startswith('<'):
				continue
			assert '>' in line, f'Malformed tess tag in {file_name}'
			file_text.write(line[line.index('>') + 1:].strip())
			file_text.write(' ')
	return file_text.getvalue()

FILE_PARSERS = {
	'tess': parse_tess,
}

def _get_filenames(corpus_dir, file_extensions, excluded_paths):
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
			if '.' in current_file_name and current_file_name[current_file_name.rindex('.') + 1:] in file_extensions \
			and join(current_path, current_file_name) not in excluded_paths:
				file_names.append(join(current_path, current_file_name))
	return sorted(file_names)

def _extract_features(corpus_dir, file_extension_to_parse_function, excluded_paths, features, output_file):
	file_names = _get_filenames(corpus_dir, file_extension_to_parse_function.keys(), excluded_paths)
	feature_tuples = [(name, textual_feature.decorated_features[name]) for name in features]
	text_to_features = {} #Associates file names to their respective features
	print(
		f'Extracting features from file with extensions '
		f'[{", ".join(file_extension_to_parse_function.keys())}] in directory {c.yellow(corpus_dir)}'
	)

	#Feature extraction
	for file_name in file_names if output_file is None else tqdm(file_names, dynamic_ncols=True):
		text_to_features[file_name] = {}
		file_extension = file_name[file_name.rindex('.') + 1:]
		file_text = file_extension_to_parse_function[file_extension](file_name)

		for feature_name, feature_func in feature_tuples:
			score = feature_func(text=file_text, filepath=file_name)
			text_to_features[file_name][feature_name] = score
			if output_file is None:
				print(f'{file_name}, {str(feature_name)}, {c.green(str(score))}')

	textual_feature.clear_cache()

	if output_file is not None:
		print(f'Feature mining complete. Attempting to write feature results to "{c.yellow(output_file)}"...')
		with open(output_file, 'wb') as pickle_file:
			pickle_file.write(pickle.dumps(text_to_features))
		print(c.green('Success!'))

# Keys of file_extension_to_parse_function must not include the dot e.g. use txt not .txt
# If excluded_paths is given, it must be a set and it can contain files or directories (the directories must
# end in a file separator e.g. slash on Mac or Linux)
#pylint: disable = too-many-branches
def main(corpus_dir, file_extension_to_parse_function, excluded_paths=None, features=None, output_file=None):
	'''Run feature extraction on all decorated features'''
	if excluded_paths is None: excluded_paths = set()
	if features is None: features = textual_feature.decorated_features.keys()

	if not corpus_dir: raise ValueError('Must provide a directory that contains the corpus')
	if not file_extension_to_parse_function or not isinstance(file_extension_to_parse_function, clctn.Mapping):
		raise ValueError('Must provide a mapping from file extensions to functions specifying how to parse them')
	if None in file_extension_to_parse_function:
		raise ValueError('The keys of file_extension_to_parse_function must not be None')
	if not all(callable(f) for f in file_extension_to_parse_function.values()):
		raise ValueError('The values of file_extension_to_parse_function must be callable')
	if any('.' in extension for extension in file_extension_to_parse_function.keys()):
		raise ValueError(
			f'The following extensions contain a period. '
			f'Please remove the periods: {list(filter(lambda x: "." in x, file_extension_to_parse_function.keys()))}'
		)
	if not features:
		raise ValueError(
			f'No features were provided. Ensure you have declared and annotated '
			f'them with the decorator @textual_feature before calling this function'
		)
	if not os.path.isdir(corpus_dir): raise ValueError(f'Path "{corpus_dir}" is not a valid directory')
	if not isinstance(excluded_paths, set): raise ValueError('Excluded paths must be in a set')
	if not all(os.path.isfile(path) or os.path.isdir(path) for path in excluded_paths):
		raise ValueError(f'Each path in {str(excluded_paths)} must be a valid path for a file or directory!')
	if not all(name in textual_feature.decorated_features.keys() for name in features):
		raise ValueError(
			f'The values in set {str(set(features) - textual_feature.decorated_features.keys())} '
			f'are not among the decorated features in {str(textual_feature.decorated_features.keys())}'
		)

	if output_file:
		if not isinstance(output_file, str): raise ValueError('Output file must be a string for a file path')
		if os.path.isfile(output_file): raise ValueError(f'Output file "{output_file}" already exists!')
		if os.path.isdir(output_file):
			raise ValueError(f'The end of the path "{output_file}" is a directory - please specify a filename')
		if os.sep in output_file and not os.path.isdir(os.path.dirname(output_file)):
			raise ValueError(f'"{os.path.dirname(output_file)}" is not a valid directory!')
	elif output_file is not None: raise ValueError('Output file must be truthy, or None')

	from timeit import timeit
	from functools import partial
	print(
		'\n\n' + c.green(
			'Feature mining elapsed time: ' + '%.4f' % timeit(
				partial(
					_extract_features, corpus_dir, file_extension_to_parse_function,
					excluded_paths, features, output_file
				),
				number=1
			) + ' seconds'
		)
	)
