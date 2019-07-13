from .textual_feature import textual_feature, decorated_features, \
	tokenize_types, clear_cache, debug_output, setup_tokenizers
import unittest

setup_tokenizers(terminal_punctuation=('.', '?')) #'FULL STOP', 'SEMICOLON', 'GREEK QUESTION MARK'

@textual_feature(tokenize_type='sentences', debug=True)
def foo(text):
	return 'foo'

@textual_feature(tokenize_type='sentences', debug=True)
def bar(text):
	return 'bar'

@textual_feature(tokenize_type='words', debug=True)
def taz(text):
	return 'taz'

@textual_feature(tokenize_type='words', debug=True)
def qux(text):
	return 'qux'

@textual_feature(tokenize_type='sentences', debug=True)
def rup(text):
	return 'rup'

@textual_feature(tokenize_type='sentences', debug=True)
def lon(text):
	return 'lon'

@textual_feature(tokenize_type='sentences', debug=True)
def return_sentences(text):
	return text

@textual_feature(tokenize_type='words', debug=True)
def return_words(text):
	return text

class TestTextualFeature(unittest.TestCase):

	def setUp(self):
		clear_cache(tokenize_types, debug_output)

	def test_cache(self):
		file = 'test test. test test test test test test? test test. test.'

		filename = 'abc/def'
		foo(text=file, filepath=filename)
		bar(text=file, filepath=filename)
		rup(text=file, filepath=filename)
		self.assertEqual(debug_output.getvalue(), 'Cache hit! function: <bar>, filepath: abc/def\n' + \
			'Cache hit! function: <rup>, filepath: abc/def\n')

		clear_cache(tokenize_types, debug_output)

		filename = 'abc/ghi'
		foo(text=file, filepath=filename)
		taz(text=file, filepath=filename)
		qux(text=file, filepath=filename)
		self.assertEqual(debug_output.getvalue(), 'Cache hit! function: <qux>, filepath: abc/ghi\n')
		filename = 'abc/jkl'
		foo(text=file, filepath=filename)
		bar(text=file, filepath=filename)
		taz(text=file, filepath=filename)
		qux(text=file, filepath=filename)
		self.assertEqual(debug_output.getvalue(), 'Cache hit! function: <qux>, filepath: abc/ghi\n' + \
			'Cache hit! function: <bar>, filepath: abc/jkl\n' + \
			'Cache hit! function: <qux>, filepath: abc/jkl\n')

	def test_sentence_tokenization(self):
		file = 'test test. test test test test test test? test test. test.'
		filename = 'abc/def'
		self.assertEqual(return_sentences(text=file, filepath=filename), ['test test.', 'test test test test test test?', \
			'test test.', 'test.'])

	def test_word_tokenization(self):
		file = 'test test. test test test test test test? test test. test.'
		filename = 'abc/def'
		self.assertEqual(return_words(text=file, filepath=filename), ['test', 'test', '.', 'test', 'test', 'test', 'test', \
			'test', 'test', '?', 'test', 'test', '.', 'test', '.'])

	def test_loop_over_all_features(self):
		file = 'test test. test test test test test test? test test. test.'
		filename = 'abc/def'
		outputs = ['foo', 'bar', 'taz', 'qux', 'rup', 'lon', ['test test.', 'test test test test test test?', \
			'test test.', 'test.'], ['test', 'test', '.', 'test', 'test', 'test', 'test', 'test', 'test', '?', 'test', \
			'test', '.', 'test', '.']]

		i = 0
		for v in decorated_features.values():
			self.assertEqual(v(text=file, filepath=filename), outputs[i])
			i += 1

	def test_no_filename(self):
		file = 'test test. test test test test test test? test test. test.'
		filename = 'abc/def'
		a = foo(text=file, filepath=filename)
		b = foo(text=file)
		self.assertEqual(a, b)

if __name__ == '__main__':
	unittest.main()
