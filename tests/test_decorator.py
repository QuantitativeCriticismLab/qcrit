#pylint: disable = missing-docstring, blacklisted-name, unused-argument, invalid-name, unexpected-keyword-arg
'''Test decorators'''
import unittest

import context #pylint: disable=unused-import
import qcrit.textual_feature

qcrit.textual_feature.setup_tokenizers(terminal_punctuation=('.', '?'))

@qcrit.textual_feature.textual_feature(tokenize_type='sentences', debug=True)
def foo(text):
	return 'foo'

@qcrit.textual_feature.textual_feature(tokenize_type='sentences', debug=True)
def bar(text):
	return 'bar'

@qcrit.textual_feature.textual_feature(tokenize_type='words', debug=True)
def taz(text):
	return 'taz'

@qcrit.textual_feature.textual_feature(tokenize_type='words', debug=True)
def qux(text):
	return 'qux'

@qcrit.textual_feature.textual_feature(tokenize_type='sentences', debug=True)
def rup(text):
	return 'rup'

@qcrit.textual_feature.textual_feature(tokenize_type='sentences', debug=True)
def lon(text):
	return 'lon'

@qcrit.textual_feature.textual_feature(tokenize_type='sentences', debug=True)
def return_sentences(text):
	return text

@qcrit.textual_feature.textual_feature(tokenize_type='words', debug=True)
def return_words(text):
	return text

class TestTextualFeature(unittest.TestCase):

	def setUp(self):
		qcrit.textual_feature.clear_cache()

	def test_cache(self):
		file = 'test test. test test test test test test? test test. test.'

		filename = 'abc/def'
		foo(text=file, filepath=filename)
		bar(text=file, filepath=filename)
		rup(text=file, filepath=filename)
		self.assertEqual(
			qcrit.textual_feature.debug_output.getvalue(), 'Cache hit! function: <bar>, filepath: abc/def\n' +
			'Cache hit! function: <rup>, filepath: abc/def\n'
		)

		self.assertEqual(qcrit.textual_feature.tokenize_types['sentences']['prev_filepath'], 'abc/def')
		self.assertEqual(qcrit.textual_feature.tokenize_types['sentences']['tokens'], ['test test.', 'test test test test test test?', 'test test.', 'test.'])

		qcrit.textual_feature.clear_cache()

		self.assertEqual(qcrit.textual_feature.tokenize_types[None]['prev_filepath'], None)
		self.assertEqual(qcrit.textual_feature.tokenize_types[None]['tokens'], None)

		filename = 'abc/ghi'
		foo(text=file, filepath=filename)
		taz(text=file, filepath=filename)
		qux(text=file, filepath=filename)
		self.assertEqual(
			qcrit.textual_feature.debug_output.getvalue(), 'Cache hit! function: <qux>, filepath: abc/ghi\n'
		)
		filename = 'abc/jkl'
		foo(text=file, filepath=filename)
		bar(text=file, filepath=filename)
		taz(text=file, filepath=filename)
		qux(text=file, filepath=filename)
		self.assertEqual(
			qcrit.textual_feature.debug_output.getvalue(), 'Cache hit! function: <qux>, filepath: abc/ghi\n' +
			'Cache hit! function: <bar>, filepath: abc/jkl\n' + \
			'Cache hit! function: <qux>, filepath: abc/jkl\n'
		)

	def test_sentence_tokenization(self):
		file = 'test test. test test test test test test? test test. test.'
		filename = 'abc/def'
		self.assertEqual(return_sentences(text=file, filepath=filename), [
			'test test.',
			'test test test test test test?',
			'test test.', 'test.'
		])

	def test_word_tokenization(self):
		file = 'test test. test test test test test test? test test. test.'
		filename = 'abc/def'
		self.assertEqual(return_words(text=file, filepath=filename), [
			'test', 'test', '.', 'test', 'test', 'test', 'test',
			'test', 'test', '?', 'test', 'test', '.', 'test', '.'
		])

	def test_loop_over_all_features(self):
		file = 'test test. test test test test test test? test test. test.'
		filename = 'abc/def'
		outputs = [
			'foo', 'bar', 'taz', 'qux', 'rup', 'lon',
			['test test.', 'test test test test test test?', 'test test.', 'test.'],
			[
				'test', 'test', '.', 'test', 'test', 'test', 'test', 'test', 'test',
				'?', 'test', 'test', '.', 'test', '.'
			]
		]

		i = 0
		for feat in qcrit.textual_feature.decorated_features.values():
			self.assertEqual(feat(text=file, filepath=filename), outputs[i])
			i += 1

	def test_no_filename(self):
		file = 'test test. test test test test test test? test test. test.'
		filename = 'abc/def'
		a = foo(text=file, filepath=filename)
		b = foo(text=file)
		self.assertEqual(a, b)

if __name__ == '__main__':
	unittest.main()
