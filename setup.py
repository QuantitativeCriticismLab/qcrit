from setuptools import setup

with open('README.md', 'r') as fh:
	long_description = fh.read()

setup(
	name='qcrit',
	version='0.0.1',
	py_modules=['qcrit'],
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://www.qcrit.org',
	author='Tim Gianitsos',
)

