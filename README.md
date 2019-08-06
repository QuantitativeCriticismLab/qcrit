Utilities for The Quantitative Criticism Lab
https://www.qcrit.org

## Installation
With `pip`:
```bash
pip install qcrit
```
With `pipenv`
```bash
pipenv install qcrit
```
## About

The qcrit package contains utilities to facilitate processing and analyzing literature.

### Feature extraction

A feature is a number that results from processing literature. Examples of features include the number of definite articles and the mean sentence length. The word "feature" can also refer to a python function that computes such a value.

To compute features, you must 1) traverse each text in a corpus, 2) parse the text into tokens, 3) write logic to calculate features, and 4) output the results to the console or to a file. Also, this will run slowly unless you 5) cache tokenized text for features that use the same tokens.

With the `textual_feature` decorator, steps (1), (2), (4), and (5) are abstracted away - you just need to implement (3) the logic to calculate each feature.

Once you have written a feature as a `python` function, label it with the decorator `textual_feature`. Your feature must have exactly one parameter which is assumed to be the parsed text of a file.
```python
from qcrit.textual_feature import textual_feature
@textual_feature()
def count_definite_article(text):
	return text.count('the')
```

The `textual_feature` module takes an argument that represents the type of tokenization.

There are four supported tokenization_types: 'sentences', 'words', 'sentence_words' and None. This tells the function in 
what format it will receive the 'text' parameter.
- If None, the function will receive the text parameter as a string. 
- If 'sentences', the function will receive the text parameter as a list of sentences, each as a string
- If 'words', the function will receive the text parameter as a list of words
- If 'sentence_words', the function will recieve the text parameter as a list of sentences, each as a list of words

```python
from functools import reduce
@textual_feature(tokenize_type='sentences')
def mean_sentence_len(text):
	sen_len = reduce(lambda cur_len, cur_sen: cur_len + len(cur_sen))
	num_sentences = len(text)
	return sen_len / num_sentences
```

Use `extract_features.main` to run all the functions labeled with the decorators and output results into a file.

corpus_dir - the directory to search for files containing texts, this will traverse all sub-directories as well

file_extension - restrict search to only files with this extension, handles parsing out of unnecessary tags, 
                 currently only supports .tess but easily extensible to xml, txt, etc.

output_file - the file to output the results into, created to be analyzed during machine learning phase

In order for sentence tokenization to work correctly, setup_tokenizers() must be set to the 
terminal punctuation marks of the language being analyzed. Make sure this is done before main() is called.

```python
from qcrit.textual_feature import setup_tokenizers
import qcrit.extract_features
setup_tokenizers(terminal_punctuation=('.', ';'))
qcrit.extract_features.main(
	corpus_dir='demo', file_extension='tess', output_file='output.pickle'
)

```
Output:
```bash
Extracting features from .tess files in demo/
100%|██████████████████████████████████████████| 4/4 [00:00<00:00,  8.67it/s]
Feature mining complete. Attempting to write feature results to "output.pickle"...
Success!


Feature mining elapsed time: 1.4919 seconds

```

### Analysis

Use the `@model_analyzer()` decorator to label functions that analyze machine learning models

Invoke `analyze_models.main('output.pickle', 'classifications.csv')` to
run all functions labeled with the `@model_analyzer()` decorator. To run only one function, include
the name of the function as the third parameter to analyze_models.main()

output.pickle: Now that the features have been extracted and output into output.pickle, we
can use machine learning models on them.

classifications.csv: The file classifications.csv contains the name of the file in the first column
and the particular classification (prose or verse) in the second column for every file in the corpus.

```python
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
	'output.pickle', 'classifications.csv'
)
```
Output:
```
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
```
## Development
To activate the virtual environment, ensure that you have pipenv installed, and run the following:
```bash
pipenv shell
pipenv install --dev
```

## Submission
The following commands will submit the package to the `Python Package Index`. It may be necessary to increment the version number in `setup.py` and to delete any previously generated `dist/` and `build/` directories.
```bash
python setup.py bdist_wheel sdist
twine upload dist/*
```
