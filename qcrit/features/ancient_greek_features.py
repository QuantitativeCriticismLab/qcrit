# pylint: disable = missing-docstring
'''
Ancient Greek features
'''
from functools import reduce
from unicodedata import normalize

from ..textual_feature import textual_feature
#Reference for normalization: https://jktauber.com/articles/python-unicode-ancient-greek/

@textual_feature(tokenize_type='sentence_words')
def freq_interrogatives(text):
	num_interrogative = 0
	interrogative_chars = {';', ';'}
	for line in text:
		num_interrogative += reduce(
			lambda cur_count, word: cur_count + 1 if word in interrogative_chars else 0, line, 0
		)
	return num_interrogative / len(text)

@textual_feature(tokenize_type='words')
def freq_conditional_markers(text):
	num_conditional_words = 0
	num_characters = 0
	conditional_words = {'εἰ', 'εἴ', 'εἲ', 'ἐάν', 'ἐὰν'}
	conditional_words = conditional_words | \
	{normalize('NFD', val) for val in conditional_words} | \
	{normalize('NFC', val) for val in conditional_words} | \
	{normalize('NFKD', val) for val in conditional_words} | \
	{normalize('NFKC', val) for val in conditional_words}

	for word in text:
		num_conditional_words += 1 if word in conditional_words else 0
		num_characters += len(word)

	return num_conditional_words / num_characters

@textual_feature(tokenize_type='words')
def freq_personal_pronouns(text):
	num_pronouns = 0
	num_characters = 0
	personal_pronouns = {
		'ἐγώ', 'ἐγὼ', 'ἐμοῦ', 'μου', 'ἐμοί', 'ἐμοὶ', 'μοι', 'ἐμέ', 'ἐμὲ', 'με', 'ἡμεῖς', 'ἡμῶν',
		'ἡμῖν', 'ἡμᾶς', 'σύ', 'σὺ', 'σοῦ', 'σου', 'σοί', 'σοὶ', 'σοι', 'σέ', 'σὲ', 'σε', 'ὑμεῖς',
		'ὑμῶν', 'ὑμῖν', 'ὑμᾶς', 'μ', 'σ'
	}
	personal_pronouns = personal_pronouns | \
	{normalize('NFD', val) for val in personal_pronouns} | \
	{normalize('NFC', val) for val in personal_pronouns} | \
	{normalize('NFKD', val) for val in personal_pronouns} | \
	{normalize('NFKC', val) for val in personal_pronouns}

	for word in text:
		num_pronouns += 1 if word in personal_pronouns else 0
		num_characters += len(word)

	return num_pronouns / num_characters

@textual_feature(tokenize_type='words')
def freq_demonstrative(text):
	num_demonstratives = 0
	num_characters = 0
	demonstrative_pronouns = {
		'ἐκεῖνος', 'ἐκείνου', 'ἐκείνῳ', 'ἐκεῖνον', 'ἐκεῖνοι', 'ἐκείνων', 'ἐκείνοις', 'ἐκείνους',
		'ἐκείνη', 'ἐκείνης', 'ἐκείνῃ', 'ἐκείνην', 'ἐκεῖναι', 'ἐκείναις', 'ἐκείνᾱς', 'ἐκείνας',
		'ἐκεῖνο', 'ἐκεῖνα', 'ὅδε', 'τοῦδε', 'τῷδε', 'τόνδε', 'οἵδε', 'τῶνδε', 'τοῖσδε', 'τούσδε',
		'ἥδε', 'τῆσδε', 'τῇδε', 'τήνδε', 'αἵδε', 'ταῖσδε', 'τᾱ́σδε', 'τάσδε', 'τόδε', 'τάδε',
		'οὗτος', 'τούτου', 'τούτῳ', 'τοῦτον', 'οὗτοι', 'τούτων', 'τούτοις', 'τούτους', 'αὕτη',
		'ταύτης', 'ταύτῃ', 'ταύτην', 'αὕται', 'ταύταις', 'ταύτᾱς', 'ταύτας', 'τοῦτο', 'ταῦτα',
		'ἐκεῖν', 'ὅδ', 'τοῦδ', 'τῷδ', 'τόνδ', 'οἵδ', 'τῶνδ', 'τοῖσδ', 'τούσδ', 'ἥδ', 'τῆσδ',
		'τῇδ', 'τήνδ', 'αἵδ', 'ταῖσδ', 'τάσδ', 'τόδ', 'τάδ'
	}
	demonstrative_pronouns = demonstrative_pronouns | \
	{normalize('NFD', val) for val in demonstrative_pronouns} | \
	{normalize('NFC', val) for val in demonstrative_pronouns} | \
	{normalize('NFKD', val) for val in demonstrative_pronouns} | \
	{normalize('NFKC', val) for val in demonstrative_pronouns}

	for word in text:
		num_demonstratives += 1 if word in demonstrative_pronouns else 0
		num_characters += len(word)

	return num_demonstratives / num_characters

@textual_feature(tokenize_type='sentence_words')
def freq_indefinite_pronoun_in_non_interrogative_sentence(text):
	num_indefinite_pronouns = 0
	num_characters = 0
	interrogative_chars = {';', ';'}
	pronoun_chars = {
		'τις', 'τινός', 'τινὸς', 'του', 'τινί', 'τινὶ', 'τῳ', 'τινά', 'τινὰ', 'τινές',
		'τινὲς', 'τινῶν', 'τισί', 'τισὶ', 'τισίν', 'τισὶν', 'τινάς', 'τινὰς', 'τι'
	}
	pronoun_chars = pronoun_chars | \
	{normalize('NFD', val) for val in pronoun_chars} | \
	{normalize('NFC', val) for val in pronoun_chars} | \
	{normalize('NFKD', val) for val in pronoun_chars} | \
	{normalize('NFKC', val) for val in pronoun_chars}

	for line in text:
		if line[-1] not in interrogative_chars and len(line) > 1 and line[-2] not in interrogative_chars:
			for word in line:
				num_indefinite_pronouns += 1 if word in pronoun_chars else 0
				num_characters += len(word)

	return num_indefinite_pronouns / num_characters

@textual_feature(tokenize_type='words')
def freq_allos(text):
	num_allos = 0
	num_characters = 0
	allos_characters = {
		'ἄλλος', 'ἄλλη', 'ἄλλο', 'ἄλλου', 'ἄλλῳ', 'ἄλλον', 'ἄλλοι', 'ἄλλων', 'ἄλλοις', 'ἄλλους',
		'ἄλλης', 'ἄλλῃ', 'ἄλλην', 'ἄλλαι', 'ἄλλᾱς', 'ἄλλας', 'ἄλλα'
	}
	allos_characters = allos_characters | \
	{normalize('NFD', val) for val in allos_characters} | \
	{normalize('NFC', val) for val in allos_characters} | \
	{normalize('NFKD', val) for val in allos_characters} | \
	{normalize('NFKC', val) for val in allos_characters}

	for word in text:
		num_allos += 1 if word in allos_characters else 0
		num_characters += len(word)

	return num_allos / num_characters

@textual_feature(tokenize_type='words')
def freq_autos(text):
	num_autos = 0
	num_characters = 0
	autos_characters = {
		'αὐτός', 'αὐτὸς', 'αὐτοῦ', 'αὐτῷ', 'αὐτόν', 'αὐτὸν', 'αὐτοί', 'αὐτοὶ', 'αὐτῶν', 'αὐτοῖς',
		'αὐτούς', 'αὐτοὺς', 'αὐτή', 'αὐτὴ', 'αὐτῆς', 'αὐτῇ', 'αὐτήν', 'αὐτὴν', 'αὐταί', 'αὐταὶ',
		'αὐταῖς', 'αὐτᾱς', 'αὐτᾱ́ς', 'αὐτάς', 'αὐτὰς', 'αὐτό', 'αὐτὸ', 'αὐτά', 'αὐτὰ'
	}
	autos_characters = autos_characters | \
	{normalize('NFD', val) for val in autos_characters} | \
	{normalize('NFC', val) for val in autos_characters} | \
	{normalize('NFKD', val) for val in autos_characters} | \
	{normalize('NFKC', val) for val in autos_characters}

	for word in text:
		num_autos += 1 if word in autos_characters else 0
		num_characters += len(word)

	return num_autos / num_characters

@textual_feature(tokenize_type='words')
def freq_reflexive(text):
	num_reflexive = 0
	num_characters = 0

	reflexive_characters = {
		'ἐμαυτοῦ', 'ἐμαυτῷ', 'ἐμαυτόν', 'ἐμαυτὸν', 'ἐμαυτῆς', 'ἐμαυτῇ', 'ἐμαυτήν', 'ἐμαυτὴν',
		'σεαυτοῦ', 'σεαυτῷ', 'σεαυτόν', 'σεαυτὸν', 'σεαυτῆς', 'σεαυτῇ', 'σεαυτήν', 'σεαυτὴν',
		'ἑαυτοῦ', 'ἑαυτῷ', 'ἑαυτόν', 'ἑαυτὸν', 'ἑαυτῶν', 'ἑαυτοῖς', 'ἑαυτούς', 'ἑαυτοὺς',
		'ἑαυτῆς', 'ἑαυτῇ', 'ἑαυτήν', 'ἑαυτὴν', 'ἑαυταῖς', 'ἑαυτάς', 'ἑαυτὰς', 'ἑαυτό', 'ἑαυτὸ',
		'ἑαυτά', 'ἑαυτὰ'
	}
	reflexive_characters = reflexive_characters | \
	{normalize('NFD', val) for val in reflexive_characters} | \
	{normalize('NFC', val) for val in reflexive_characters} | \
	{normalize('NFKD', val) for val in reflexive_characters} | \
	{normalize('NFKC', val) for val in reflexive_characters}

	bigram_reflexive_characters = {
		'ἡμῶν': {'αὐτῶν'}, 'ἡμῖν': {'αὐτοῖς', 'αὐταῖς'},
		'ἡμᾶς': {'αὐτούς', 'αὐτοὺς', 'αὐτάς', 'αὐτὰς'}, 'ὑμῶν': {'αὐτῶν'}, 'ὑμῖν': {'αὐτοῖς', 'αὐταῖς'},
		'ὑμᾶς': {'αὐτούς', 'αὐτοὺς', 'αὐτάς', 'αὐτὰς'}, 'σφῶν': {'αὐτῶν'}, 'σφίσιν': {'αὐτοῖς', 'αὐταῖς'},
		'σφᾶς': {'αὐτούς', 'αὐτοὺς', 'αὐτάς', 'αὐτὰς'}
	}
	#This is just verbose syntax for normalizing all the keys and values
	#in the dictionary with NFD, NFC, NFKD, & NFKC. The double star (**) unpacking is
	#how dictionaries are merged https://stackoverflow.com/a/26853961/7102572
	bigram_reflexive_characters = {
		**bigram_reflexive_characters,
		**{
			normalize('NFD', key): {normalize('NFD', v) for v in val}
			for key, val in bigram_reflexive_characters.items()
		},
		**{
			normalize('NFC', key): {normalize('NFC', v) for v in val}
			for key, val in bigram_reflexive_characters.items()
		},
		**{
			normalize('NFKD', key): {normalize('NFKD', v) for v in val}
			for key, val in bigram_reflexive_characters.items()
		},
		**{
			normalize('NFKC', key): {normalize('NFKC', v) for v in val}
			for key, val in bigram_reflexive_characters.items()
		},
	}

	bigram_first_half = None
	for word in text:

		#Found monogram characters
		if word in reflexive_characters:
			num_reflexive += 1
			bigram_first_half = None

		#Found the first part of the reflexive bigram
		elif word in bigram_reflexive_characters:
			bigram_first_half = word

		#Found the second part of the reflexive bigram
		elif bigram_first_half in bigram_reflexive_characters \
				and word in bigram_reflexive_characters[bigram_first_half]:
			num_reflexive += 2
			bigram_first_half = None

		#Default case
		else:
			bigram_first_half = None

		num_characters += len(word)

	return num_reflexive / num_characters

@textual_feature(tokenize_type='sentence_words')
def freq_sentences_with_vocative_omega(text):
	num_vocatives = 0
	vocative_characters = {'ὦ'}
	vocative_characters = vocative_characters | \
	{normalize('NFD', val) for val in vocative_characters} | \
	{normalize('NFC', val) for val in vocative_characters} | \
	{normalize('NFKD', val) for val in vocative_characters} | \
	{normalize('NFKC', val) for val in vocative_characters}

	for line in text:
		for word in line:
			if word in vocative_characters:
				num_vocatives += 1
				break

	return num_vocatives / len(text)

@textual_feature(tokenize_type='words')
def freq_superlative(text):
	num_superlative = 0
	num_characters = 0
	superlative_ending_characters = [
		'τατος', 'τάτου', 'τάτῳ', 'τατον', 'τατοι', 'τάτων',
		'τάτοις', 'τάτους', 'τάτη', 'τάτης', 'τάτῃ', 'τάτην',
		'τάταις', 'τάτας', 'τατα', 'τατά', 'τατε'
	]
	#The endswith() method requires a tuple
	superlative_ending_characters = tuple(superlative_ending_characters + \
	[normalize('NFD', val) for val in superlative_ending_characters] + \
	[normalize('NFC', val) for val in superlative_ending_characters] + \
	[normalize('NFKD', val) for val in superlative_ending_characters] + \
	[normalize('NFKC', val) for val in superlative_ending_characters])

	for word in text:
		num_superlative += 1 if word.endswith(superlative_ending_characters) else 0
		num_characters += len(word)

	return num_superlative / num_characters

@textual_feature(tokenize_type='words')
def freq_conjunction(text):
	num_conjunction = 0
	num_characters = 0
	conjunction_chars = {
		'τε', 'καί', 'καὶ', 'ἀλλά', 'ἀλλὰ', 'καίτοι', 'οὐδέ', 'οὐδὲ', 'μηδέ', 'μηδὲ', 'οὔτε',
		'οὔτ', 'μήτε', 'μήτ', 'οὐδ', 'μηδ', 'ἤ', 'ἢ', 'τ'
	}
	conjunction_chars = conjunction_chars | \
	{normalize('NFD', val) for val in conjunction_chars} | \
	{normalize('NFC', val) for val in conjunction_chars} | \
	{normalize('NFKD', val) for val in conjunction_chars} | \
	{normalize('NFKC', val) for val in conjunction_chars}

	for word in text:
		num_conjunction += 1 if word in conjunction_chars else 0
		num_characters += len(word)

	return num_conjunction / num_characters

@textual_feature(tokenize_type='sentence_words')
def mean_sentence_length(text):
	return reduce(lambda cur_len, line: cur_len + reduce(
		lambda word_len, word: word_len + len(word), line, 0
	), text, 0) / len(text)

@textual_feature(tokenize_type='sentence_words')
def freq_sentence_with_relative_clause(text):
	num_sentence_with_clause = 0
	num_sentences = 0
	pronouns = {
		'ὅς', 'ὃς', 'οὗ', 'ᾧ', 'ὅν', 'ὃν', 'οἵ', 'οἳ', 'ὧν', 'οἷς', 'οὕς', 'οὓς', 'ἥ',
		'ἣ', 'ἧς', 'ᾗ', 'ἥν', 'ἣν', 'αἵ', 'αἳ', 'αἷς', 'ἅς', 'ἃς', 'ὅ', 'ὃ', 'ἅ', 'ἃ'
	}
	pronouns = pronouns | \
	{normalize('NFD', val) for val in pronouns} | \
	{normalize('NFC', val) for val in pronouns} | \
	{normalize('NFKD', val) for val in pronouns} | \
	{normalize('NFKC', val) for val in pronouns}

	for line in text:
		for word in line:
			if word in pronouns:
				num_sentence_with_clause += 1
				break
		num_sentences += 1

	return num_sentence_with_clause / num_sentences

@textual_feature(tokenize_type='words')
def mean_length_relative_clause(text):
	num_relative_clause = 0
	sum_length_relative_clause = 0
	pronouns = {
		'ὅς', 'ὃς', 'οὗ', 'ᾧ', 'ὅν', 'ὃν', 'οἵ', 'οἳ', 'ὧν', 'οἷς', 'οὕς', 'οὓς', 'ἥ',
		'ἣ', 'ἧς', 'ᾗ', 'ἥν', 'ἣν', 'αἵ', 'αἳ', 'αἷς', 'ἅς', 'ἃς', 'ὅ', 'ὃ', 'ἅ', 'ἃ'
	}
	pronouns = pronouns | \
	{normalize('NFD', val) for val in pronouns} | \
	{normalize('NFC', val) for val in pronouns} | \
	{normalize('NFKD', val) for val in pronouns} | \
	{normalize('NFKC', val) for val in pronouns}
	punctuation = {'.', ',', ':', ';', ';'}
	punctuation = punctuation | \
	{normalize('NFD', val) for val in punctuation} | \
	{normalize('NFC', val) for val in punctuation} | \
	{normalize('NFKD', val) for val in punctuation} | \
	{normalize('NFKC', val) for val in punctuation}

	in_relative_clause = False

	for word in text:
		if word in punctuation:
			in_relative_clause = False
		elif word in pronouns:
			in_relative_clause = True
			num_relative_clause += 1
		if in_relative_clause:
			sum_length_relative_clause += len(word)

	return 0 if num_relative_clause == 0 else sum_length_relative_clause / num_relative_clause

@textual_feature(tokenize_type='words')
def freq_circumstantial_markers(text):
	num_participles = 0
	num_characters = 0
	participles = {'ἔπειτα', 'ὅμως', 'καίπερ', 'ἅτε', 'ἔπειτ', 'ἅτ', 'ὁμῶς'}
	participles = participles | \
	{normalize('NFD', val) for val in participles} | \
	{normalize('NFC', val) for val in participles} | \
	{normalize('NFKD', val) for val in participles} | \
	{normalize('NFKC', val) for val in participles}

	for word in text:
		num_participles += 1 if word in participles else 0
		num_characters += len(word)

	return num_participles / num_characters

@textual_feature(tokenize_type='words')
def freq_hina(text):
	num_hina = 0
	num_characters = 0
	ina_characters = {'ἵνα', 'ἵν'}
	ina_characters = ina_characters | \
	{normalize('NFD', val) for val in ina_characters} | \
	{normalize('NFC', val) for val in ina_characters} | \
	{normalize('NFKD', val) for val in ina_characters} | \
	{normalize('NFKC', val) for val in ina_characters}

	for word in text:
		num_hina += 1 if word in ina_characters else 0
		num_characters += len(word)

	return num_hina / num_characters

@textual_feature(tokenize_type='words')
def freq_hopos(text):
	num_hopos = 0
	num_characters = 0
	hopos_characters = {'ὅπως'}
	hopos_characters = hopos_characters | \
	{normalize('NFD', val) for val in hopos_characters} | \
	{normalize('NFC', val) for val in hopos_characters} | \
	{normalize('NFKD', val) for val in hopos_characters} | \
	{normalize('NFKC', val) for val in hopos_characters}

	for word in text:
		num_hopos += 1 if word in hopos_characters else 0
		num_characters += len(word)

	return num_hopos / num_characters

@textual_feature(tokenize_type='words')
def freq_ws(text):
	num_ws = 0
	num_characters = 0
	ws_characters = {'ὡς'}
	ws_characters = ws_characters | \
	{normalize('NFD', val) for val in ws_characters} | \
	{normalize('NFC', val) for val in ws_characters} | \
	{normalize('NFKD', val) for val in ws_characters} | \
	{normalize('NFKC', val) for val in ws_characters}

	for word in text:
		num_ws += 1 if word in ws_characters else 0
		num_characters += len(word)

	return num_ws / num_characters

@textual_feature(tokenize_type='words')
def freq_wste_not_preceded_by_eta(text):
	num_wste = 0
	num_characters = 0
	wste_characters = {'ὥστε'}
	wste_characters = wste_characters | \
	{normalize('NFD', val) for val in wste_characters} | \
	{normalize('NFC', val) for val in wste_characters} | \
	{normalize('NFKD', val) for val in wste_characters} | \
	{normalize('NFKC', val) for val in wste_characters}
	eta_chars = {'ἤ', 'ἢ'}
	eta_chars = eta_chars | \
	{normalize('NFD', val) for val in eta_chars} | \
	{normalize('NFC', val) for val in eta_chars} | \
	{normalize('NFKD', val) for val in eta_chars} | \
	{normalize('NFKC', val) for val in eta_chars}
	ok_to_add = True

	for word in text:
		num_wste += 1 if word in wste_characters and ok_to_add else 0
		num_characters += len(word)
		ok_to_add = word not in eta_chars

	return num_wste / num_characters

@textual_feature(tokenize_type='words')
def freq_temporal_causal_markers(text):
	num_clause_words = 0
	num_characters = 0
	clause_chars = {
		'μέϰρι', 'ἕως', 'πρίν', 'πρὶν', 'ἐπεί', 'ἐπεὶ', 'ἐπειδή',
		'ἐπειδὴ', 'ἐπειδάν', 'ἐπειδὰν', 'ὅτε', 'ὅταν'
	}
	clause_chars = clause_chars | \
	{normalize('NFD', val) for val in clause_chars} | \
	{normalize('NFC', val) for val in clause_chars} | \
	{normalize('NFKD', val) for val in clause_chars} | \
	{normalize('NFKC', val) for val in clause_chars}

	for word in text:
		num_clause_words += 1 if word in clause_chars else 0
		num_characters += len(word)

	return num_clause_words / num_characters

@textual_feature(tokenize_type='sentence_words')
def variance_of_sentence_length(text):
	num_sentences = 0
	total_len = 0

	for line in text:
		num_sentences += 1
		total_len += reduce(lambda cur_len, word: cur_len + len(word), line, 0)
	mean = total_len / num_sentences
	squared_difference = 0
	for line in text:
		squared_difference += (
			reduce(lambda cur_len, word: cur_len + len(word), line, 0) - mean
		) ** 2

	return squared_difference / num_sentences

@textual_feature(tokenize_type='words')
def freq_particles(text):
	num_particles = 0
	num_characters = 0
	#Word tokenizer doesn't work well with ellision - apostrophes are removed
	particles = {
		'ἄν', 'ἂν', 'ἆρα', 'γε', "γ", "δ", 'δέ', 'δὲ', 'δή', 'δὴ', 'ἕως', "κ", 'κε',
		'κέ', 'κὲ', 'κέν', 'κὲν', 'κεν', 'μά', 'μὰ', 'μέν', 'μὲν', 'μέντοι', 'μήν',
		'μὴν', 'μῶν', 'νύ', 'νὺ', 'νυ', 'οὖν', 'περ', 'πω', 'τοι'
	}
	particles = particles | \
	{normalize('NFD', val) for val in particles} | \
	{normalize('NFC', val) for val in particles} | \
	{normalize('NFKD', val) for val in particles} | \
	{normalize('NFKC', val) for val in particles}

	for word in text:
		num_particles += 1 if word in particles else 0
		num_characters += len(word)

	return num_particles / num_characters

@textual_feature(tokenize_type='words')
def freq_men(text):
	num_men = 0
	num_characters = 0
	men_chars = {'μέν', 'μὲν'}
	men_chars = men_chars | \
	{normalize('NFD', val) for val in men_chars} | \
	{normalize('NFC', val) for val in men_chars} | \
	{normalize('NFKD', val) for val in men_chars} | \
	{normalize('NFKC', val) for val in men_chars}

	for word in text:
		num_men += 1 if word in men_chars else 0
		num_characters += len(word)
	return num_men / num_characters
