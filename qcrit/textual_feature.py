#pylint: disable = protected-access, invalid-name, unnecessary-lambda
'''Utilities for textual feature decorator'''
import re
from inspect import signature
from collections import OrderedDict
from io import StringIO

import nltk.tokenize.punkt as punkt

decorated_features = OrderedDict()
lang = None
word_tokenizer = None
sentence_tokenizers = None
debug_output = StringIO()
NON_WORD_CHARS = (
	r"\?¿؟\!¡！‽…⋯᠁ฯ,،，､、。°※··᛫~\:;;\\\/⧸⁄（）\(\)\[\]\{\}\<\>"
	r"\'\"‘’“”‹›«»《》\|‖\=\-\‐\‒\–\—\―_\+\*\^\$£€§%#@&†‡"
)

tokenize_types = {
	None: {
		'func': lambda text: text,
		'prev_filepath': None,
		'tokens': None,
	},
	'sentences': {
		'func': lambda text: sentence_tokenizers[lang].tokenize(text),
		'prev_filepath': None,
		'tokens': None,
	},
	'words': {
		'func': lambda text: word_tokenizer.word_tokenize(text),
		'prev_filepath': None,
		'tokens': None,
	},
	'sentence_words': {
		'func': lambda text: [word_tokenizer.word_tokenize(s) for s in sentence_tokenizers[lang].tokenize(text)],
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

def setup_tokenizers(*, language=None, terminal_punctuation):
	'''Initialize the word tokenizer and sentence tokenizer given the terminal punctuation'''
	global lang
	global word_tokenizer
	global sentence_tokenizers
	global tokenize_types
	lang = language #TODO validate in this function, not in textual_feature()
	punkt.PunktLanguageVars.sent_end_chars = terminal_punctuation
	punkt.PunktLanguageVars.re_boundary_realignment = re.compile(r'[›»》’”\'\"）\)\]\}\>]+?(?:\s+|(?=--)|$)', re.MULTILINE)
	clear_cache()

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

	sentence_tokenizers = {None: punkt.PunktSentenceTokenizer(lang_vars=punkt.PunktLanguageVars())}
	for sen_tkzr in sentence_tokenizers.values():
		sen_tkzr._lang_vars._re_period_context = sent_tok_vars._re_period_context
		sen_tkzr._lang_vars._re_word_tokenizer = sent_tok_vars._re_word_tokenizer

def textual_feature(*, tokenize_type=None, debug=False):
	'''Decorator for textual features'''
	global lang
	if not (word_tokenizer and sentence_tokenizers):
		raise ValueError(
			f'Tokenizers not initialized: Use '
			f'"setup_tokenizers(terminal_punctuation=<collection of punctutation>)" before decorating a function'
		)
	if tokenize_type not in tokenize_types:
		raise ValueError(
			'"' + str(tokenize_type) + '" is not a valid tokenize type: Choose from among ' +
			str(list(tokenize_types.keys()))
		)
	if lang not in sentence_tokenizers:
		raise ValueError(
			f'"{str(lang)}" is not an available language. Choose from among '
			f'{list(sentence_tokenizers.keys())}'
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
