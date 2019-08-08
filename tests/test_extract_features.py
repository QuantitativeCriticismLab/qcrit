#pylint: disable = missing-docstring, blacklisted-name, unused-argument, invalid-name
'''Test feature extraction'''
import unittest

import context #pylint: disable=unused-import
from qcrit.extract_features import main, parse_tess
from qcrit.textual_feature import textual_feature, setup_tokenizers

#Run this file with "-b" to ignore output in passing tests (failing tests still display output)

setup_tokenizers(terminal_punctuation=('.', ';', ';')) #'FULL STOP', 'SEMICOLON', 'GREEK QUESTION MARK'

@textual_feature(tokenize_type='words', debug=True)
def dummy_feature(text):
	pass

class TestExtractFeatures(unittest.TestCase):

	def testAllNone(self):
		self.assertRaises(ValueError, main, corpus_dir=None, file_extension_to_parse_function=None)

	def testInvalidCorpusDirectory(self):
		self.assertRaises(ValueError, main, corpus_dir='abc', file_extension_to_parse_function={'tess': parse_tess})

	def testExcludedPaths(self):
		self.assertRaises(ValueError, main, corpus_dir='.', file_extension_to_parse_function={'tess': parse_tess}, excluded_paths=[])

	def testEmptyFeatures(self):
		self.assertRaises(ValueError, main, corpus_dir='.', file_extension_to_parse_function={'tess': parse_tess}, features=[])

	def testOutputAlreadyExists(self):
		self.assertRaises(ValueError, main, corpus_dir='.', file_extension_to_parse_function={'tess': parse_tess}, output_file=__file__)

	def testOutputDirectoryDoesntExist(self):
		self.assertRaises(ValueError, main, corpus_dir='.', file_extension_to_parse_function={'tess': parse_tess}, output_file='a/b')

	def testOutputFalsy(self):
		self.assertRaises(ValueError, main, corpus_dir='.', file_extension_to_parse_function={'tess': parse_tess}, output_file='')

	def testOutputDirectoryInvalidAndNoFile(self):
		self.assertRaises(ValueError, main, corpus_dir='.', file_extension_to_parse_function={'tess': parse_tess}, output_file='a/')

	def testOutputDirectoryValidAndNoFile(self):
		self.assertRaises(ValueError, main, corpus_dir='.', file_extension_to_parse_function={'tess': parse_tess}, output_file='./')

	def testOutputDirectoryValidAndNoFile2(self):
		self.assertRaises(ValueError, main, corpus_dir='.', file_extension_to_parse_function={'tess': parse_tess}, output_file='.')

if __name__ == '__main__':
	unittest.main()
