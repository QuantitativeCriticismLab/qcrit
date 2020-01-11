# pylint: disable = missing-docstring
'''
Language-independent features
'''

import re
from statistics import mean, StatisticsError
from collections import Counter
from string import punctuation
from math import nan

from qcrit.textual_feature import textual_feature

_WORD_REGEX = re.compile(r'^\w+$')
_DEGENERATE_PLACEHOLDER = '@'
assert not _WORD_REGEX.match(_DEGENERATE_PLACEHOLDER) and len(_DEGENERATE_PLACEHOLDER) >= 1

@textual_feature(tokenize_type='sentence_words')
def average_sentence_length(text):
	try:
		return mean(sum(len(word) for word in sentence) for sentence in text)
	except StatisticsError:
		return 0

@textual_feature(tokenize_type='words')
def ratio_capital_to_lowercase(text):
	capital_cnt = sum(1 for word in text for letter in word if letter.isupper())
	lowercase_cnt = sum(1 for word in text for letter in word if letter.islower())
	return capital_cnt / lowercase_cnt if lowercase_cnt else nan

@textual_feature(tokenize_type='words')
def ratio_lowercase_to_totalchars(text):
	lowercase_cnt = sum(1 for word in text for letter in word if letter.islower())
	total_cnt = sum(len(word) for word in text)
	return lowercase_cnt / total_cnt if total_cnt else 0

@textual_feature(tokenize_type=None)
def ratio_punctuation_to_spaces(text):
	punctuation_cnt = sum(text.count(punc) for punc in f'{punctuation}‘’“”') #include slant quotes
	space_cnt = text.count(' ')
	return punctuation_cnt / space_cnt if space_cnt else nan

@textual_feature(tokenize_type='words')
def mean_word_length(text):
	try:
		return mean(len(word) for word in text if _WORD_REGEX.match(word))
	except StatisticsError:
		return 0

def _sentence_boundary_counter_helper(generator, sentence_cnt):
	counts = Counter(generator)
	sent_cnt_with_only_non_words = counts[_DEGENERATE_PLACEHOLDER]
	del counts[_DEGENERATE_PLACEHOLDER] #remove placeholder for sentences with only non-words
	return (
		counts.most_common(1)[0][1] / (sentence_cnt - sent_cnt_with_only_non_words)
		if sentence_cnt - sent_cnt_with_only_non_words else 0
	)

@textual_feature(tokenize_type='sentence_words')
def freq_most_frequent_start_word(text):
	return _sentence_boundary_counter_helper(
		(
			#Search forward for first word token in sentence
			next((word for word in sentence if _WORD_REGEX.match(word)), _DEGENERATE_PLACEHOLDER)
			for sentence in text
		),
		len(text)
	)

@textual_feature(tokenize_type='sentence_words')
def freq_most_frequent_stop_word(text):
	return _sentence_boundary_counter_helper(
		(
			#Search backward for last word token in sentence
			next((word for word in reversed(sentence) if _WORD_REGEX.match(word)), _DEGENERATE_PLACEHOLDER)
			for sentence in text
		),
		len(text)
	)

@textual_feature(tokenize_type='sentence_words')
def freq_most_frequent_start_letter_start_word(text):
	return _sentence_boundary_counter_helper(
		(
			next((word for word in sentence if _WORD_REGEX.match(word)), _DEGENERATE_PLACEHOLDER)[0].lower()
			for sentence in text
		),
		len(text)
	)

@textual_feature(tokenize_type='sentence_words')
def freq_most_frequent_start_letter_stop_word(text):
	return _sentence_boundary_counter_helper(
		(
			next((
				word for word in reversed(sentence) if _WORD_REGEX.match(word)
			), _DEGENERATE_PLACEHOLDER)[0].lower()
			for sentence in text
		),
		len(text)
	)

@textual_feature(tokenize_type='words')
def freq_single_occurrence_words(text):
	counts = Counter(word for word in text if _WORD_REGEX.match(word))
	total = sum(counts.values())
	return sum(freq for _, freq in counts.items() if freq == 1) / total if total else 0

@textual_feature(tokenize_type='words')
def freq_double_occurrence_words(text):
	counts = Counter(word for word in text if _WORD_REGEX.match(word))
	total = sum(counts.values())
	return sum(freq for _, freq in counts.items() if freq == 2) / total if total else 0

@textual_feature(tokenize_type='words')
def freq_words_with_word_length_three(text):
	counts = Counter(
		'invalid_word_placeholder' if not _WORD_REGEX.match(word) else
		'valid_word_invalid_len_placeholder' if len(word) != 3 else
		'valid_word_valid_len_placeholder' for word in text
	)
	valid_word_cnt = len(text) - counts['invalid_word_placeholder']
	return counts['valid_word_valid_len_placeholder'] / valid_word_cnt if valid_word_cnt else 0

@textual_feature(tokenize_type='words')
def freq_words_in_interval_word_length_4_6_inclusive(text):
	counts = Counter(
		'invalid_word_placeholder' if not _WORD_REGEX.match(word) else
		'valid_word_invalid_len_placeholder' if not 4 <= len(word) <= 6 else
		'valid_word_valid_len_placeholder' for word in text
	)
	valid_word_cnt = len(text) - counts['invalid_word_placeholder']
	return counts['valid_word_valid_len_placeholder'] / valid_word_cnt if valid_word_cnt else 0
