Easily extract features from texts, and run machine learning algorithms on them. Write your own features, use ours, or do both!

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

To get started, just replace `'your-directory-name'` with the name of a directory of `.txt` files. Everything else is taken care of!
```python
from qcrit.extract_features import main
from qcrit.textual_feature import setup_tokenizers
import qcrit.features.universal_features
setup_tokenizers(terminal_punctuation=('.', '!', '?'))
main(
	corpus_dir='your-directory-name', file_extension_to_parse_function={'txt': lambda filename: open(filename).read()}
)
```

### Writing Your Own Features

A feature is a number that results from processing literature. An example of a feature might be the number of definite articles, the mean sentence length, or the fraction of interrogative sentences. The word "feature" can also refer to a python function that computes such a value.

Normally to compute features, you must 1) obtain a corpus of texts, 2) traverse each text in the corpus, 3) parse the text into tokens, 4) write logic to calculate features, and 5) output the results to the console or to a file. Also, this will run slowly unless you 6) cache tokenized text for features that use the same tokens.

With the `textual_feature` decorator, steps (2), (3), (4), (5), and (6) are abstracted away - you just need (1) to supply the corpus.

Once you have written a feature as a `python` function, label it with the decorator `textual_feature`. Your feature must have exactly one parameter which is assumed to be the parsed text of a file.
```python
from qcrit.textual_feature import textual_feature
@textual_feature()
def count_definite_article(text):
	return text.count('the')
```

The `textual_feature` decorator takes an argument that represents the type of tokenization.

There are four supported tokenization_types: `'sentences'`, `'words'`, `'sentence_words'` and `None`. This tells the function in 
what format it will receive the `'text'` parameter.
- If `None`, the function will receive the text parameter as a string. 
- If `'sentences'`, the function will receive the text parameter as a list of sentences, each as a string
- If `'words'`, the function will receive the text parameter as a list of words
- If `'sentence_words'`, the function will recieve the text parameter as a list of sentences, each as a list of words

```python
from functools import reduce
@textual_feature(tokenize_type='sentences')
def mean_sentence_len(text):
	sen_len = reduce(lambda cur_len, cur_sen: cur_len + len(cur_sen), text, 0)
	num_sentences = len(text)
	return sen_len / num_sentences
```

### Extracting Features

Use `qcrit.extract_features.main` to run all the functions labeled with the decorators and output results into a file.

`corpus_dir` - the directory to search for files containing texts, this will traverse all sub-directories as well

`file_extension_to_parse_function` - map from file extension (e.g. 'txt', 'tess') of texts that you would like to parse to a function directing how to parse it

`output_file` - the file to output the results into, created to be analyzed during machine learning phase

In order for sentence tokenization to work correctly, `setup_tokenizers()` must be called with the 
terminal punctuation marks of the language being analyzed. You can also optionally supply the name of the language as well. If data exists about how to parse the language, this may improve sentence tokenization.

```python
from qcrit.extract_features import main, parse_tess
from qcrit.textual_feature import setup_tokenizers
from somewhere_else import count_definite_article, mean_sentence_len
setup_tokenizers(terminal_punctuation=('.', '!', '?'), language='greek')
main(
	corpus_dir='demo', file_extension_to_parse_function={'tess': parse_tess}, output_file='output.pickle'
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
1. Ensure that you have `pipenv` installed. Also, ensure that you have a version of `python` installed that matches the version in the `Pipfile`.
1. Setup a virtual environment and install the necessary dependencies:
	```bash
	PIPENV_VENV_IN_PROJECT=true pipenv install --dev
	```
1. Activate the virtual environment:
	```bash
	pipenv shell
	```
	Now, `python` commands will use the dependencies and `python` version from the virtual environment. Use `exit` to leave the virtual environment, and use `pipenv shell` while in the project directory to activate it again.

### Demo
```bash
python demo/demo.py
```

## Submission
The following commands will submit the package to the `Python Package Index`. Before running them, it may be necessary to increment the version number in `__init__.py` and to delete any previously generated `dist/`, `build/`, and `egg-info` directories if they exist.
```bash
python setup.py bdist_wheel sdist
twine upload dist/*
```
