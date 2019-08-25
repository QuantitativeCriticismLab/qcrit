'''
ANSI Colors
'''
def red(input_str):
	'''Output red'''
	return '\033[91m' + input_str + '\033[0m'
def green(input_str):
	'''Output green'''
	return '\033[92m' + input_str + '\033[0m'
def yellow(input_str):
	'''Output yellow'''
	return '\033[93m' + input_str + '\033[0m'
def purple(input_str):
	'''Output purple'''
	return '\033[95m' + input_str + '\033[0m'
