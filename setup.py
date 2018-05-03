# -*- coding: utf-8 -*-
""" setup

Package setup for kpext

"""

from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

setup(name='python-kpext',
      version='v0.1.0.dev3',
      description='Python package for keyphrase extraction.',
      long_description=README,
      long_description_content_type='text/markdown',
      url='https://github.com/snovd/keyphrase-extraction',
      author='Simon D. Hernandez',
      author_email='py.kpext@totum.one',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      keywords='keyphrase extraction',
      packages=find_packages('src'),
      package_dir={'':'src', 'kpext': 'src/kpext'},
      package_data={'kpext': ['kpext_data/models/*']},
      python_requires='>=3, <4',
      platform='any',
      install_requires=['nltk', 'python-crfsuite'],
      zip_safe=False)
