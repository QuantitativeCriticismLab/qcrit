# From https://stackoverflow.com/a/34325723

_prev_str_length = None

# Print iterations progress
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	"""
	Call in a loop to create terminal progress bar
	@params:
	    iteration   - Required  : current iteration (Int)
	    total       - Required  : total iterations (Int)
	    prefix      - Optional  : prefix string (Str)
	    suffix      - Optional  : suffix string (Str)
	    decimals    - Optional  : positive number of decimals in percent complete (Int)
	    length      - Optional  : character length of bar (Int)
	    fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	s = '%s |%s| %s%% %s' % (prefix, bar, percent, suffix)
	global _prev_str_length
	if _prev_str_length:
		print(' ' * _prev_str_length, end='\r') #Clear out previous bar to prevent lingering characters if current bar is shorter
	print(s, end='\r')
	_prev_str_length = len(s)

	# Print New Line on Complete
	if iteration == total: 
		_prev_str_length = None
		print()

if __name__ == '__main__':
	# 
	# Sample Usage
	# 

	from time import sleep

	#get width (columns) of terminal
	import shutil;
	columns, rows = shutil.get_terminal_size()

	# A List of Items
	items = list(range(0, 57))
	l = len(items)	#57
	barLen = int(columns) - 27 # prefix, bar, percent, suffix are 27 spaces

	#if bar length is too small, don't print bar
	if barLen >= 0:
		# Initial call to print 0% progress
		print_progress_bar(0, l, prefix = 'Progress:', suffix = 'Complete', length = barLen)
		for i, item in enumerate(items):
			# Do stuff...
			sleep(0.1)
			# Update Progress bar
			print_progress_bar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = barLen)


	else :
		print("Window size too small to show progress bar")

	# Sample Output
	# Progress: |█████████████████████████████████████████████-----| 90.0% Complete







