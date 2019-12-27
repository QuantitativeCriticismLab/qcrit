'''
Setup
'''
import re
from os.path import join

import setuptools
import pipfile

setuptools.setup(
	name='qcrit',
	version=re.search(
		#Parse the version from the text of __init__.py because
		#importing it directly before setup can cause problems
		#with the setup itself
		r'__version__\s*=\s*[\'"](\d+\.\d+\.\d+)[\'"]',
		open(join('qcrit', '__init__.py'), mode='r').read()
	).group(1),
	author='Tim Gianitsos',
	author_email='contact@qcrit.org',
	description='Quantitative Criticism Lab',
	long_description=open('README.md', mode='r').read(),
	long_description_content_type='text/markdown',
	url='https://www.qcrit.org',
	project_urls={'Source Code': 'https://github.com/QuantitativeCriticismLab/qcrit'},
	packages=setuptools.find_packages(),
	classifiers=[
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
	],
	install_requires=[
		pkg + (version if version != '*' else '')
		for pkg, version in pipfile.load('Pipfile').data['default'].items()
	],
)
