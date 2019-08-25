#pylint: disable = wrong-import-order, wrong-import-position, missing-docstring, unused-argument
#Not tested on windows

import os

_CURRENT_DIR = os.path.dirname(__file__)
#If the output file already exists, the feature extraction code will not override it
#Delete the output file so that the demo can create one
if os.path.isfile(os.path.join(_CURRENT_DIR, 'output.pickle')):
	os.system('rm ' + os.path.join(_CURRENT_DIR, 'output.pickle'))

import context #pylint: disable=unused-import
import qcrit.extract_features
from qcrit.textual_feature import textual_feature, setup_tokenizers
from functools import reduce
from unicodedata import normalize

#Let sentence tokenizer know that periods and semicolons are the punctuation marks that end sentences
setup_tokenizers(terminal_punctuation=('.', ';'))

#Using 'words' makes the input 'text' parameter become a list of words
@textual_feature(tokenize_type='words')
def num_conjunctions(text): #parameter must be the text of a file
	return reduce(
		lambda count, word: count + (
			1 if word in {normalize('NFD', val) for val in ['καί', 'καὶ', 'ἀλλά', 'ἀλλὰ', 'ἤ', 'ἢ']} else 0
		), text, 0
	)

#Using 'sentences' makes the input 'text' parameter become a list of sentences
@textual_feature(tokenize_type='sentences')
def mean_sentence_length(text): #parameter must be the text of a file
	return reduce(lambda count, sentence: count + len(sentence), text, 0) / len(text)

#Not putting any decorator parameters will leave the input 'text' parameter unchanged as a string of text
@textual_feature()
def num_interrogatives(text): #parameter must be the text of a file
	return text.count(';')

qcrit.extract_features.main(
	corpus_dir=_CURRENT_DIR,
	file_extension_to_parse_function={'tess': qcrit.extract_features.parse_tess},
	output_file=os.path.join(_CURRENT_DIR, 'output.pickle')
)

import qcrit.analyze_models
from qcrit.model_analyzer import model_analyzer
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@model_analyzer()
def feature_rankings(data, target, file_names, feature_names, labels_key):
	print('-' * 40 + '\nRandom Forest Classifier feature rankings\n')
	features_train, features_test, labels_train, _ = train_test_split(data, target, test_size=0.5, random_state=0)
	clf = ensemble.RandomForestClassifier(random_state=0, n_estimators=10)
	clf.fit(features_train, labels_train)
	clf.predict(features_test)

	#Display features in order of importance
	print('Feature importances:')
	for tup in sorted(zip(feature_names, clf.feature_importances_), key=lambda s: -s[1]):
		print('\t%f: %s' % (tup[1], tup[0]))

@model_analyzer()
def classifier_accuracy(data, target, file_names, feature_names, labels_key):
	print('-' * 40 + '\nRandom Forest Classifier accuracy\n')
	features_train, features_test, labels_train, labels_test = train_test_split(
		data, target, test_size=0.5, random_state=0
	)
	clf = ensemble.RandomForestClassifier(random_state=0, n_estimators=10)
	clf.fit(features_train, labels_train)
	results = clf.predict(features_test)

	print('Stats:')
	print(
		'\tNumber correct: ' + str(accuracy_score(labels_test, results, normalize=False)) +
		' / ' + str(len(results))
	)
	print('\tPercentage correct: ' + str(accuracy_score(labels_test, results) * 100) + '%')

@model_analyzer()
def misclassified_texts(data, target, file_names, feature_names, labels_key):
	print('-' * 40 + '\nRandom Forest Classifier misclassified texts\n')
	features_train, features_test, labels_train, labels_test, idx_train, idx_test = train_test_split(
		data, target, range(len(target)), test_size=0.5, random_state=0
	)
	print('Train texts:\n\t' + '\n\t'.join(file_names[i] for i in idx_train) + '\n')
	print('Test texts:\n\t' + '\n\t'.join(file_names[i] for i in idx_test) + '\n')
	clf = ensemble.RandomForestClassifier(random_state=0, n_estimators=10)
	clf.fit(features_train, labels_train)
	results = clf.predict(features_test)

	print('Misclassifications:')
	for i, _ in enumerate(results):
		if results[i] != labels_test[i]:
			print('\t' + file_names[idx_test[i]])

qcrit.analyze_models.main(
	os.path.join(_CURRENT_DIR, 'output.pickle'), os.path.join(_CURRENT_DIR, 'classifications.csv')
)
