#pylint: disable = protected-access, invalid-name, unnecessary-lambda
'''Utilities for textual feature decorator'''
import re
from inspect import signature
from collections import OrderedDict
from io import StringIO
import os
from os.path import join, dirname, isdir, isfile, abspath, lexists
import sys
import pickle

import nltk
import nltk.tokenize.punkt as punkt

decorated_features = OrderedDict()
word_tokenizer = None
sentence_tokenizer = None
debug_output = StringIO()
NON_WORD_CHARS = (
	r"\?¿؟\!¡！‽…⋯᠁ฯ,،，､、。°※··᛫~\:;;\\\/⧸⁄（）\(\)\[\]\{\}\<\>"
	r"\'\"‘’“”`‹›«»《》\|‖\=\-\‐\‒\–\—\―_\+\*\^\$£€§%#@&†‡"
)

tokenize_types = {
	None: {
		'func': lambda text: text,
		'prev_filepath': None,
		'tokens': None,
	},
	'sentences': {
		'func': lambda text: sentence_tokenizer.tokenize(text),
		'prev_filepath': None,
		'tokens': None,
	},
	'words': {
		'func': lambda text: word_tokenizer.word_tokenize(text),
		'prev_filepath': None,
		'tokens': None,
	},
	'sentence_words': {
		'func': lambda text: [word_tokenizer.word_tokenize(s) for s in sentence_tokenizer.tokenize(text)],
		'prev_filepath': None,
		'tokens': None,
	},
}

def clear_cache():
	'''Clear tokens from previously parsed texts'''
	global tokenize_types
	global debug_output
	for _, val in tokenize_types.items():
		val['prev_filepath'] = None
		val['tokens'] = None
	debug_output.truncate(0)
	debug_output.seek(0)

def setup_tokenizers(*, terminal_punctuation, language=None):
	'''Initialize the word tokenizer and sentence tokenizer given the terminal punctuation'''
	global word_tokenizer
	global sentence_tokenizer
	global tokenize_types
	if word_tokenizer or sentence_tokenizer:
		raise Exception('Tokenizers have already been initialized')

	clear_cache()
	punkt.PunktLanguageVars.sent_end_chars = terminal_punctuation
	punkt.PunktLanguageVars.re_boundary_realignment = re.compile(
		r'[›»》’”\'\"）\)\]\}\>]+?(?:\s+|(?=--)|$)', re.MULTILINE
	)

	'''
	Accessing private variables of punkt.PunktLanguageVars because
	nltk has a faulty design pattern that necessitates it.
	Issue reported here: https://github.com/nltk/nltk/issues/2068
	'''

	'''
	A word tokenizer should strip the non word chars from words,
	as well as periods and numbers
	'''
	word_tokenizer = punkt.PunktLanguageVars()
	word_tokenizer._re_word_tokenizer = re.compile(punkt.PunktLanguageVars._word_tokenize_fmt % {
		'NonWord': fr"(?:[\d\.{NON_WORD_CHARS}])",
		'MultiChar': punkt.PunktLanguageVars._re_multi_char_punct,
		'WordStart': fr"[^\d\.{NON_WORD_CHARS}]",
	}, re.UNICODE | re.VERBOSE)
	word_tokenizer._re_period_context = re.compile(punkt.PunktLanguageVars._period_context_fmt % {
		'NonWord': fr"(?:[\d\.{NON_WORD_CHARS}])",
		'SentEndChars': word_tokenizer._re_sent_end_chars,
	}, re.UNICODE | re.VERBOSE)

	'''
	A sentence tokenizer should strip the non word chars from words.
	This regex excludes periods because the original regex in the Punkt class
	excludes them, and it excludes numbers because we do not want to
	treat numbers with decimals as if they were sentences
	'''
	sent_tok_vars = punkt.PunktLanguageVars()
	sent_tok_vars._re_word_tokenizer = re.compile(punkt.PunktLanguageVars._word_tokenize_fmt % {
		'NonWord': fr"(?:[{NON_WORD_CHARS}])",
		'MultiChar': punkt.PunktLanguageVars._re_multi_char_punct,
		'WordStart': fr"[^{NON_WORD_CHARS}]",
	}, re.UNICODE | re.VERBOSE)
	sent_tok_vars._re_period_context = re.compile(punkt.PunktLanguageVars._period_context_fmt % {
		'NonWord': fr"(?:[{NON_WORD_CHARS}])",
		'SentEndChars': sent_tok_vars._re_sent_end_chars,
	}, re.UNICODE | re.VERBOSE)

	if language:
		#Attempt to download language-specific pretrained sentence tokenizer models from nltk
		#Assume that the directory name that was downloaded from running
		#`nltk.download('punkt')` will always be named 'tokenizers'
		nltk_punkt_dir = join(dirname(__file__), 'tokenizers')
		if not lexists(nltk_punkt_dir):
			print('Attempting to download language-specific sentence tokenizer models from nltk...')
			try:
				nltk.download(info_or_id='punkt', download_dir=dirname(__file__), raise_on_error=True)
			except Exception as e:
				print(
					'Failed to download sentence tokenization language data.'
					' Consider leaving the language unspecified. This may cause sentence tokenization to'
					' not properly recognize abbreviations, but otherwise it has reasonable performance.',
					file=sys.stderr
				)
				raise e
			print(
				f'Successfully downloaded tokenizer models to '
				f'{join(abspath(dirname(__file__)), "tokenizers")}'
			)

		#Attempt to load nltk data
		models_dir = join(nltk_punkt_dir, 'punkt', 'PY3')
		if not isdir(models_dir):
			import errno
			print(
				'NLTK language data may not have been downloaded correctly.'
				f' Consider leaving the language unspecified, or delete {abspath(nltk_punkt_dir)}'
				' if it exists, and try again.',
				file=sys.stderr)
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), models_dir)
		try:
			sentence_tokenizer = pickle.load(open(join(
				models_dir, f'{language}{os.extsep}pickle'
			), mode='rb'))
		except Exception as e:
			sep = '", "'
			print(
				f'Unable to load language data for "{language}"\nAvailable languages: '
				f'''"{
					sep.join(
						['None'] + [m[:m.index(os.extsep)] for m in os.listdir(models_dir)
						if m.endswith(f"{os.extsep}pickle") and isfile(join(models_dir, m))]
					)
				}"''',
				file=sys.stderr
			)
			raise e
	else:
		sentence_tokenizer = punkt.PunktSentenceTokenizer(lang_vars=punkt.PunktLanguageVars())

	sentence_tokenizer._lang_vars._re_period_context = sent_tok_vars._re_period_context
	sentence_tokenizer._lang_vars._re_word_tokenizer = sent_tok_vars._re_word_tokenizer

def textual_feature(*, tokenize_type=None, debug=False):
	'''Decorator for textual features'''
	if tokenize_type not in tokenize_types:
		raise ValueError(
			'"' + str(tokenize_type) + '" is not a valid tokenize type: Choose from among ' +
			str(list(tokenize_types.keys()))
		)
	def decor(f):
		#TODO make this more extensible. Use keyword args somehow instead of 'text' parameter?
		#TODO Ensure that features with duplicated names aren’t put in the ordered dict (this can happen if they come from different files)
		reqrd_params = {'text'}
		sig_params = signature(f).parameters
		if not all(tok in sig_params for tok in reqrd_params):
			raise ValueError(
				f'Error for feature "{f.__name__}"'
				f'\nMinimal required parameters: {str(reqrd_params)}'
				f'\nFound parameters: {set(sig_params) if sig_params else "{}"}'
			)
		def wrapper(*, text, filepath=None):
			if not word_tokenizer or not sentence_tokenizer:
				raise ValueError(
					f'Tokenizers not initialized: Use'
					f' "setup_tokenizers(terminal_punctuation=<tuple of punctutation>)"'
					f' before running functions'
				)
			if not filepath:
				return f(tokenize_types[tokenize_type]['func'](text))

			#Cache the tokenized version of this text if this filepath is new
			if tokenize_types[tokenize_type]['prev_filepath'] != filepath:
				tokenize_types[tokenize_type]['prev_filepath'] = filepath
				tokenize_types[tokenize_type]['tokens'] = tokenize_types[tokenize_type]['func'](text)
			elif debug:
				debug_output.write('Cache hit! ' + 'function: <' + f.__name__ + '>, filepath: ' + filepath + '\n')
			return f(tokenize_types[tokenize_type]['tokens'])
		decorated_features[f.__name__] = wrapper
		return wrapper
	return decor
