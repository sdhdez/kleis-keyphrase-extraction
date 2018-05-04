# -*- coding: utf-8 -*-
""" setup

Package setup for kpext

"""

from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

setup(name='kleis-keyphrase-extraction',
      version='v0.1.1.dev1',
      description='Python package for keyphrase labeling.',
      long_description=README,
      long_description_content_type='text/markdown',
      url='https://github.com/sdhdez/kleis-keyphrase-extraction',
      author='Simon D. Hernandez',
      author_email='py.kleis@totum.one',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      keywords='keyphrase-extraction keyphrase-labeling keyword-extraction crf pos-tag-sequences',
      packages=find_packages('src'),
      package_dir={'':'src', 'kleis': 'src/kleis'},
      package_data={'kleis': ['kleis_data/models/*', 'kleis_data/train/*']},
      python_requires='>=3.5, <4',
      platform='any',
      install_requires=['nltk', 'python-crfsuite'],
      zip_safe=False)
