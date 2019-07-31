#pylint: disable = wrong-import-order, wrong-import-position, missing-docstring, unused-argument
#Not tested on windows

#******************************************************************************************************************
'''
0) Setup
'''

import os

_CURRENT_DIR = os.path.dirname(__file__)
#If the output file already exists, the feature extraction code will not override it
#Delete the output file so that the demo can create one
if os.path.isfile(os.path.join(_CURRENT_DIR, 'output.pickle')):
	os.system('rm ' + os.path.join(_CURRENT_DIR, 'output.pickle'))

#******************************************************************************************************************
'''
1) Extract the features

A feature is some quantitative measure extracted by analyzing the text in a file. 
Examples of features include the mean sentence length of a file or the frequency of conjunctions in a file.

Use the python decorator textual_feature to create a feature.

@textual_feature()
def foo(file): #must have one parameter which is assumed be the parsed text of a file
	return 0

The textual_feature decorator takes two optional arguments: the type of tokenization, and the language.

@textual_feature(tokenize_type='sentences', lang='ancient_greek')
def bar(file):
	return 0

There are four supported tokenization_types: 'sentences', 'words', 'sentence_words' and None. This tells the function in 
what format it will receive the 'file' parameter.
- If None, the function will receive the file parameter as a string. 
- If 'sentences', the function will receive the file parameter as a list of sentences, each as a string
- If 'words', the function will receive the file parameter as a list of words
- If 'sentence_words', the function will recieve the file parameter as a list of sentences, each as a list of words

There are several supported lang options: None, 'greek', 'estonian', 'ancient_greek', 'turkish', 'polish', 
'czech', 'portuguese', 'dutch', 'norwegian', 'slovene', 'english', 'danish', 'finnish', 'swedish', 
'spanish', 'german', 'italian', 'french'

If the language needed is not in this list, using None will often suffice. The lang option is used only 
in sentence tokenization. It's purpose is to help the sentence tokenizer identify abbreviations in a 
given language (so that it doesn't mistake an abbreviation as a sentence ending). If there are very few 
abbreviation in a corpus, the utility of specifying the language will be marginal.

Use extract_features.main to run all the functions labeled with the decorators, and output results into a file

corpus_dir - the directory to search for files containing texts, this will traverse all sub-directories as well
file_extension - restrict search to only files with this extension, handles parsing out of unnecessary tags, 
                 currently only supports .tess but easily extensible to xml, txt, etc.
output_file - the file to output the results into, created to be analyzed during machine learning phase

In order for sentence tokenization to work correctly, setup_tokenizers() must be set to the 
terminal punctuation marks of the language being analyzed. Make sure this is done before main() is called.
'''
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
	corpus_dir=_CURRENT_DIR, file_extension='tess', output_file=os.path.join(_CURRENT_DIR, 'output.pickle')
)
'''
Extracting features from .tess files in .
Progress |███████████████████████████████████████████| 100.0% (4 of 4 files)
Feature mining complete. Attempting to write feature results to "./output.pickle"...
Success!


Feature mining elapsed time: 1.4919 seconds
'''

#******************************************************************************************************************
'''
2) Train & Test machine learning models on the features

Use the "@model_analyzer()" decorator to label functions that analyze machine learning models

Invoke "analyze_models.main('demo_files/output.pickle', 'demo_files/classifications.csv')" to
run all functions labeled with the "@model_analyzer()" decorator. To run only one function, include
the name of the function as the third parameter to analyze_models.main()

output.pickle: Now that the features have been extracted and output into output.pickle, we
can use machine learning models on them.

classifications.csv: The file classifications.csv contains the name of the file in the first column
and the particular classification (prose or verse) in the second column for every file in the corpus.
'''
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
'''
----------------------------------------
Random Forest Classifier feature rankings

Feature importances:
	0.400000: num_conjunctions
	0.400000: num_interrogatives
	0.200000: mean_sentence_length


Elapsed time: 0.0122 seconds

----------------------------------------
Random Forest Classifier accuracy

Stats:
	Number correct: 1 / 2
	Percentage correct: 50.0%


Elapsed time: 0.0085 seconds

----------------------------------------
Random Forest Classifier misclassified texts

Train texts:
	demo/aristotle.poetics.tess
	demo/aristophanes.ecclesiazusae.tess

Test texts:
	demo/euripides.heracles.tess
	demo/plato.respublica.part.1.tess

Misclassifications:
	demo/plato.respublica.part.1.tess


Elapsed time: 0.0082 seconds
'''
