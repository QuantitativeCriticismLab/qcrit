#pylint: disable = wrong-import-position, unused-import
'''
Providing a 'context' file allows importing from packages in
different directories. This obviates the need to keep the demo
files in the same directory as the python package
https://docs.python-guide.org/writing/structure/#test-suite
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
