'''
Setup
'''
import setuptools

with open('README.md', 'r') as fh:
	LONG_DESCRIPTION = fh.read()

setuptools.setup(
	name='qcrit',
	version='0.0.9',
	author='Tim Gianitsos',
	author_email='contact@qcrit.org',
	description='Quantitative Criticism Lab',
	long_description=LONG_DESCRIPTION,
	long_description_content_type='text/markdown',
	url='https://www.qcrit.org',
	packages=setuptools.find_packages(),
	classifiers=[
		'Programming Language :: Python :: 3.6',
		'License :: OSI Approved :: MIT License',
		"Operating System :: OS Independent",
	],
)
