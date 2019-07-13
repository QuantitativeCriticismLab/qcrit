# -*- coding: utf-8 -*-
import unittest
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars
import re

from . import textual_feature

#[^\s\d’”\'\"）\)\]\}\.,:;]
#[“‘—\-†&vâ\*\^（α-ωΑ-Ὠ`̔]
#΄´´``′″‴
textual_feature.setup_tokenizers(terminal_punctuation=('.', ';', ';'))
p = PunktLanguageVars()
#TODO don't mess with the PunktLanguageVars instance variables, mess with the class variables
p._re_word_tokenizer = re.compile(PunktLanguageVars._word_tokenize_fmt % {
    'NonWord': r"(?:[\d\.\?¿؟\!¡！‽…⋯᠁ฯ,،，､、。°※··᛫~\:;;\\\/⧸⁄（）\(\)\[\]\{\}\<\>\'\"‘’“”‹›«»《》\|‖\=\-\‐\‒\–\—\―_\+\*\^\$£€§%#@&†‡])",
    'MultiChar': PunktLanguageVars._re_multi_char_punct,
    'WordStart': r"[^\d\.\?¿؟\!¡！‽…⋯᠁ฯ,،，､、。°※··᛫~\:;;\\\/⧸⁄（）\(\)\[\]\{\}\<\>\'\"‘’“”‹›«»《》\|‖\=\-\‐\‒\–\—\―_\+\*\^\$£€§%#@&†‡]",
}, re.UNICODE | re.VERBOSE)
p._re_period_context = re.compile(PunktLanguageVars._period_context_fmt % {
	'NonWord': r"(?:[\d\.\?¿؟\!¡！‽…⋯᠁ฯ,،，､、。°※··᛫~\:;;\\\/⧸⁄（）\(\)\[\]\{\}\<\>\'\"‘’“”‹›«»《》\|‖\=\-\‐\‒\–\—\―_\+\*\^\$£€§%#@&†‡])",
	'SentEndChars': p._re_sent_end_chars,
}, re.UNICODE | re.VERBOSE)
sentence_tokenizer = PunktSentenceTokenizer(lang_vars=p)

class TestParsers(unittest.TestCase):

	def setUp(self):
		pass

	def test_sentences1(self):
		file = 'test test. test test test? test test test; test test. test.'
		result = textual_feature.tokenize_types['sentences']['func'](file)
		expected = ['test test.', 'test test test? test test test;', 'test test.', 'test.']
		self.assertEqual(expected, result)

	def test_sentence_words1(self):
		file = 'test test. test test test? test test test; test test. test.'
		result = textual_feature.tokenize_types['sentence_words']['func'](file)
		expected = [['test', 'test', '.'], ['test', 'test', 'test', '?', 'test', 'test', 'test', ';'], 
		['test', 'test', '.'], ['test', '.']]
		self.assertEqual(expected, result)

	def test_sentence_words2(self):
		file = 'a b ccccccc. aaa aa bb; bb; ads ofiihwio; freino. daieof; frinoe.'
		result = textual_feature.tokenize_types['sentence_words']['func'](file)
		expected = [['a', 'b', 'ccccccc', '.'], ['aaa', 'aa', 'bb', ';'], ['bb', ';'], ['ads', 'ofiihwio', ';'], 
		['freino', '.'], ['daieof', ';'], ['frinoe', '.']]
		self.assertEqual(expected, result)

	def test_sentence_words3(self):
		file = 'a b ccccccc. aaa aa bb; bb; ads ofiihwio; freino. daieof; frinoe.'
		result = textual_feature.tokenize_types['words']['func'](file)
		expected = ['a', 'b', 'ccccccc', '.', 'aaa', 'aa', 'bb', ';', 'bb', ';', 'ads', 'ofiihwio', ';', 
		'freino', '.', 'daieof', ';', 'frinoe', '.']
		self.assertEqual(expected, result)

	def test_sentence_words4(self):
		file = 'a b ccccccc. aaa aa bb; bb; ads ofiihwio; freino. daieof; frinoe.'
		result = p.word_tokenize(file)
		expected = ['a', 'b', 'ccccccc', '.', 'aaa', 'aa', 'bb', ';', 'bb', ';', 'ads', 'ofiihwio', ';', 
		'freino', '.', 'daieof', ';', 'frinoe', '.']
		self.assertEqual(expected, result)

	def test_sentence_slant_quote(self):
		s = 'a b c. "a b c". a b c. "a b c." a b c. “a b c”. a b c. “a b c.” a b c.'
		result = textual_feature.tokenize_types['sentences']['func'](s)
		expected = ['a b c.', '"a b c".', 'a b c.', '"a b c."', 'a b c.', '“a b c”.', 'a b c.', '“a b c.”', 'a b c.']
		self.assertEqual(expected, result)

	def test_sentence_slant_quote1_5(self):
		s = 'a b c. "a b c". a b c. "a b c." a b c. “a b c”. a b c. “a b c.” a b c.'
		result = textual_feature.sentence_tokenizers[None].tokenize(s)
		expected = ['a b c.', '"a b c".', 'a b c.', '"a b c."', 'a b c.', '“a b c”.', 'a b c.', '“a b c.”', 'a b c.']
		self.assertEqual(expected, result)

	def test_sentence_slant_quote2(self):
		s = 'a b c. "a b c". a b c. "a b c." a b c. “a b c”. a b c. “a b c.” a b c.'
		result = sentence_tokenizer.tokenize(s)
		expected = ['a b c.', '"a b c".', 'a b c.', '"a b c."', 'a b c.', '“a b c”.', 'a b c.', '“a b c.”', 'a b c.']
		self.assertEqual(expected, result)

	def test_apollodorus_slant_quote(self):
		s = "καὶ εὑρέθησαν οὕτω. Μόψος δὲ συὸς οὔσης ἐπιτόκου ἠρώτα Κάλχαντα, πόσους χοίρους κατὰ γαστρὸς ἔχει καὶ πότε τέκοι:τοῦ δὲ εἰπόντος: “ὀκτώ,” μειδιάσας ὁ Μόψος ἔφη: “Κάλχας τῆς ἀκριβοῦς μαντείας ἀπεναντιῶς διακεῖται, ἐγὼ δ' ̓Απόλλωνος καὶ Μαντοῦς παῖς ὑπάρχων τῆς ἀκριβοῦς μαντείας τὴν ὀξυδορκίαν πάντως πλουτῶ, καὶ οὐχ ὡς ὁ Κάλχας ὀκτώ, ἀλλ' ἐννέα κατὰ γαστρός, καὶ τούτους ἄρρενας ὅλους ἔχειν μαντεύομαι, καὶ αὔριον ἀνυπερθέτως ἐν ἕκτῃ ὥρᾳ τεχθήσεσθαι.”ὧν γενομένων Κάλχας ἀθυμήσας ἀπέθανεκαὶ ἐτάφη ἐν Νοτίῳ."
		result = textual_feature.tokenize_types['sentences']['func'](s)
		expected = ['καὶ εὑρέθησαν οὕτω.', "Μόψος δὲ συὸς οὔσης ἐπιτόκου ἠρώτα Κάλχαντα, πόσους χοίρους κατὰ γαστρὸς ἔχει καὶ πότε τέκοι:τοῦ δὲ εἰπόντος: “ὀκτώ,” μειδιάσας ὁ Μόψος ἔφη: “Κάλχας τῆς ἀκριβοῦς μαντείας ἀπεναντιῶς διακεῖται, ἐγὼ δ' ̓Απόλλωνος καὶ Μαντοῦς παῖς ὑπάρχων τῆς ἀκριβοῦς μαντείας τὴν ὀξυδορκίαν πάντως πλουτῶ, καὶ οὐχ ὡς ὁ Κάλχας ὀκτώ, ἀλλ' ἐννέα κατὰ γαστρός, καὶ τούτους ἄρρενας ὅλους ἔχειν μαντεύομαι, καὶ αὔριον ἀνυπερθέτως ἐν ἕκτῃ ὥρᾳ τεχθήσεσθαι.", "”ὧν γενομένων Κάλχας ἀθυμήσας ἀπέθανεκαὶ ἐτάφη ἐν Νοτίῳ."]
		self.assertEqual(expected, result)
	
	def test_numbers(self):
		s = '1234.4321 32. 4324 4321 432 1. 134 52.653 142 41. 41268.'
		result = textual_feature.tokenize_types['sentences']['func'](s)
		expected = ['1234.4321 32.', '4324 4321 432 1.', '134 52.653 142 41.', '41268.']
		self.assertEqual(expected, result)

	def test_dagger(self):
		s = 'a b† c. "a b‡ c". a b c. "a b c†." a b c. “a b c†”. a b c. “a‡ b c.” a b c.'
		result = textual_feature.word_tokenizer.word_tokenize(s)
		expected = ['a', 'b', '†', 'c', '.', '"', 'a', 'b', '‡', 'c', '"', '.', 'a', 'b', 'c', '.', '"', 'a', 'b', 'c', '†', '.', '"', 'a', 'b', 'c', '.', '“', 'a', 'b', 'c', '†', '”', '.', 'a', 'b', 'c', '.', '“', 'a', '‡', 'b', 'c', '.', '”', 'a', 'b', 'c', '.']
		self.assertEqual(expected, result)

'''
#Plutarch Camillus
"οὐ μὴν π.,ρῆκεν αὐτῷ τὴν ἀρχὴν ὁ δῆμος, ἀλλὰ  βοῶν μήτε ἱππεύοντος αὐτοῦ μήτε ὁπλομαχοῦντος ἐν τοῖς ἀγῶσι δεῖσθαι, βουλευομένου δὲ μόνον καί προστάττοντος, ἠνάγκασεν ὑποστῆναι τὴν στρατηγίαν καί μεθ' ἑνὸς τῶν συναρχόντων Λευκίου Φουρίου τὸν στρατὸν ἄγειν εὐθὺς ἐπὶ τοὺς πολεμίους."

http://www.perseus.tufts.edu/hopper/text?doc=Perseus%3Atext%3A2008.01.0086%3Achapter%3D37%3Asection%3D2
http://www.perseus.tufts.edu/hopper/text?doc=Perseus%3Atext%3A2008.01.0012%3Achapter%3D37%3Asection%3D2
'''

if __name__ == '__main__':
	unittest.main()
	# print(textual_feature.sentence_tokenizers['ancient_greek']._lang_vars._re_word_tokenizer)
	# print(sentence_tokenizer._lang_vars._re_word_tokenizer)
