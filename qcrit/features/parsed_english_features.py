# pylint: disable = missing-docstring
'''
Parsed English Features
'''
import re
from statistics import mean, pvariance
from ..textual_feature import textual_feature

_TERMINAL_PUNCTUATION = ('.', '!', '?')

@textual_feature()
def pronouns(text):
	return sum(1 for _ in re.finditer(r'\bPRP\s', text))

@textual_feature()
def determiners(text):
	return sum(1 for _ in re.finditer(r'\bDT\s', text))

@textual_feature()
def suma(text):
	return sum(1 for _ in re.finditer(r'\bsome\)', text, flags=re.IGNORECASE))

@textual_feature()
def reflexives(text):
	return sum(1 for _ in re.finditer(r'(self|selves)\)', text, flags=re.IGNORECASE))

@textual_feature()
def ilca(text):
	return sum(1 for _ in re.finditer(r'\bthe\b.+?\bsame\)', text, flags=re.IGNORECASE))

@textual_feature()
def othr(text):
	return sum(1 for _ in re.finditer(r'\bother\)', text, flags=re.IGNORECASE))

@textual_feature()
def conjunct(text):
	conjs = [
		'and', 'both', 'bothe', 'boyth', 'but', 'butt', 'either', 'eyther', 'nor',
		'nother', 'neither', 'neyther', 'ne', 'or', 'ore', '&'
	]
	return sum(1 for conj in conjs for _ in re.finditer(fr'\b{conj}\)', text, flags=re.IGNORECASE))

@textual_feature()
def relclausesentences(text):
	cnt = 0
	punc = re.compile(' [' + ''.join('\\' + p for p in _TERMINAL_PUNCTUATION) + ']')
	rel = re.compile(r'\bSBAR\s+\(WHNP\b')
	match = rel.search(text, 0)
	while match:
		cnt += 1
		# start search after next terminal punctuation mark
		next_sentence = punc.search(text, match.span()[1])
		match = rel.search(text, next_sentence.span()[1]) if next_sentence else None
	return cnt

@textual_feature()
def avgrelclause(text):
	currenttext = text.split('\n')
	#from Ski's implementation
	relswitch=0
	relcount=0
	rellengths=[]
	relleng=0;
	for jj in range(0,len(currenttext)-1):
		ii=currenttext[jj];
		ii2=currenttext[jj+1];
		if 'SBAR' in str(ii) and 'WHNP' in str(ii2):
			relswitch=1;
		if relswitch==1:
			wordz=re.findall(r' [A-Za-z&+-]*\)',ii)
			for wordd in wordz:
				relleng=relleng+len(wordd.strip(")").replace('+','').replace('&','ond'))-1;
			if '(,' in ii or '(.' in ii or '(:' in ii:
				relswitch=0;
				rellengths.append(relleng)
				relcount+=1
				relleng=0;
	return mean(rellengths) if rellengths else 0

@textual_feature()
def gif(text):
	return sum(1 for _ in re.finditer(r'\bif\)', text, flags=re.IGNORECASE))

@textual_feature()
def temporalcausal(text):
	tempcaus = [
		'since', 'sens', 'sithence', 'syns', 'then', 'thenne', 'finally',
		'although', 'althoughe', 'despite', 'because', 'consequently', 'therefore',
		'thus', 'lest', 'leste', 'when', 'whenne'
	]
	return sum(1 for temp in tempcaus for _ in re.finditer(fr'\b{temp}\)', text, flags=re.IGNORECASE))

@textual_feature()
def meansent(text):
	punc = ''.join('\\' + p for p in _TERMINAL_PUNCTUATION)
	sen_lengths = []
	cur_sen_len = 0
	for match in re.finditer(fr' ([A-Za-z&+{punc}-]+?)\)', text):
		word = match.group(1)
		cur_sen_len += len(word)
		if word in _TERMINAL_PUNCTUATION:
			sen_lengths.append(cur_sen_len)
			cur_sen_len = 0
	return mean(sen_lengths)

@textual_feature()
def varsent(text):
	punc = ''.join('\\' + p for p in _TERMINAL_PUNCTUATION)
	sen_lengths = []
	cur_sen_len = 0
	for match in re.finditer(fr' ([A-Za-z&+{punc}-]+?)\)', text):
		word = match.group(1)
		cur_sen_len += len(word)
		if word in _TERMINAL_PUNCTUATION:
			sen_lengths.append(cur_sen_len)
			cur_sen_len = 0
	return pvariance(sen_lengths)

@textual_feature()
def interrogatives(text):
	return sum(1 for _ in re.finditer(r'\s\?\)', text, flags=re.IGNORECASE))

@textual_feature()
def prepositions(text):
	return sum(1 for _ in re.finditer(r'\bIN\s', text))

@textual_feature()
def superlatives(text):
	return sum(1 for _ in re.finditer(r'\bJJS\s', text))

@textual_feature()
def exclams(text):
	ex = ['o', 'lo', 'alas',]
	return sum(1 for temp in ex for _ in re.finditer(fr'\b{temp}\)', text, flags=re.IGNORECASE))

@textual_feature()
def modals(text):
	return sum(1 for _ in re.finditer(r'\bMD\s', text))

@textual_feature()
def discoursemarkers(text):
	punc = ''.join('\\' + p for p in _TERMINAL_PUNCTUATION)
	sentences = []
	cur_sentence = []
	for match in re.finditer(fr' ([A-Za-z&+{punc}-]+?)\)', text):
		word = match.group(1)
		cur_sentence.append(word)
		if word in _TERMINAL_PUNCTUATION:
			sentences.append(cur_sentence)
			cur_sentence = []

	markers = (
		('moreover',),
		('since',),
		('in', 'conclusion'),
		('first', 'of', 'all'),
		('in', 'the', 'end'),
		('to', 'the', 'extent', 'that'),
		('on', 'the', 'other', 'hand'),
		('at', 'the', 'end', 'of', 'the', 'day'),
	)
	cnt = 0
	for cur_sentence in sentences:
		for i in range(len(cur_sentence)):
			for marker_list in markers:
				satisfied = True
				for j, marker in enumerate(marker_list):
					if i + j >= len(cur_sentence) or cur_sentence[i + j].lower() != marker.lower():
						satisfied = False
						break
				cnt += 1 if satisfied else 0
	return cnt

@textual_feature()
def num_sentences(text):
	return sum(1 for punc in _TERMINAL_PUNCTUATION for _ in re.finditer(' \\' + punc + r'\)', text))

@textual_feature()
def oe_subjunctives(text):
	subjs = ('AXDS', 'AXP', 'BEDS', 'BEPS', 'HVDS', 'HVPS', 'VBDS', 'VBPH')
	return sum(1 for subj in subjs for _ in re.finditer(fr'\b{subj}\s', text))

@textual_feature()
def oe_thahwile(text):
	punc = ''.join('\\' + p for p in _TERMINAL_PUNCTUATION)
	sentences = []
	cur_sentence = []
	for match in re.finditer(fr' ([A-Za-z&+{punc}-]+?)\)', text):
		word = match.group(1)
		cur_sentence.append(word)
		if word in _TERMINAL_PUNCTUATION:
			sentences.append(cur_sentence)
			cur_sentence = []
	cnt = 0
	for sentence in sentences:
		for i in range(len(sentence) - 1):
			if sentence[i] == '+ta' and 'hwil' in sentence[i + 1]:
				cnt += 1
	return cnt

@textual_feature()
def oe_aertham(text):
	punc = ''.join('\\' + p for p in _TERMINAL_PUNCTUATION)
	sentences = []
	cur_sentence = []
	for match in re.finditer(fr' ([A-Za-z&+{punc}-]+?)\)', text):
		word = match.group(1)
		cur_sentence.append(word)
		if word in _TERMINAL_PUNCTUATION:
			sentences.append(cur_sentence)
			cur_sentence = []
	cnt = 0
	for sentence in sentences:
		for i in range(len(sentence) - 1):
			if sentence[i] == '+ar' and ('+tam' in sentence[i + 1] or '+ton' in sentence[i + 1]):
				cnt += 1
	return cnt
