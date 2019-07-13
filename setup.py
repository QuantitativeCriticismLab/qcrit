import setuptools

with open('README.md', 'r') as fh:
	long_description = fh.read()

setuptools.setup(
	name='qcrit',
	version='0.0.8',
	author='Tim Gianitsos',
	author_email='contact@qcrit.org',
	description="Quantitative Criticism Lab",
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://www.qcrit.org',
	packages=setuptools.find_packages(),
	classifiers=[
		'Programming Language :: Python :: 3.6',
		'License :: OSI Approved :: MIT License',
		"Operating System :: OS Independent",
	],
)

