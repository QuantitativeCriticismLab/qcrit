'''Decorator for functions analyzing models'''

from collections import OrderedDict

DECORATED_ANALYZERS = OrderedDict()

def model_analyzer():
	'''Decorator for functions analyzing models'''
	def decor(dec_func):
		def wrapper(data, target, file_names, feature_names, labels_key):
			return dec_func(data, target, file_names, feature_names, labels_key)
		DECORATED_ANALYZERS[dec_func.__name__] = wrapper
		return wrapper
	return decor
