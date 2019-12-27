'''
Setup
'''
import setuptools
import pipfile

setuptools.setup(
	name='qcrit',
	version='0.0.14',
	author='Tim Gianitsos',
	author_email='contact@qcrit.org',
	description='Quantitative Criticism Lab',
	long_description=open('README.md', 'r').read(),
	long_description_content_type='text/markdown',
	url='https://www.qcrit.org',
	project_urls={'Source Code': 'https://github.com/QuantitativeCriticismLab/qcrit'},
	packages=setuptools.find_packages(),
	classifiers=[
		'Programming Language :: Python :: 3.6',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
	],
	install_requires=[
		pkg + (version if version != '*' else '')
		for pkg, version in pipfile.load('Pipfile').data['default'].items()
	],
)
