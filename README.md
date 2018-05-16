# Kleis: Python package for keyphrase extraction

Kleis is a python package to label keyphrases in scientific text. It is named after the ancient greek word [κλείς](https://en.wiktionary.org/w/index.php?title=%CE%BA%CE%BB%CE%B5%CE%AF%CF%82).
## Install 

### Pip (Easy and quick)

```
$ pip install kleis-keyphrase-extraction
```

### Make your own wheel

```
$ git clone https://github.com/sdhdez/kleis-keyphrase-extraction.git
$ cd kleis-keyphrase-extraction/
$ python setup.py sdist bdist_wheel
$ pip install dist/kleis_keyphrase_extraction-0.1.X.devX-py3-none-any.whl
```
Replace X with the corresponding values.

Note: This method doesn't include pre-trained models, you should download the corpus so it can train.

## Usage 

Example [here](https://github.com/sdhdez/kleis-keyphrase-extraction/blob/r0.1.2/notebooks/minimal-example.ipynb)


## Datasets

Thepackage already includes some pre-trained models but if you want to test by your own you should download the datasets. 

Download from [SemEval 2017 Task 10](https://scienceie.github.io/resources.html) and decompress in "~/kleis_data/corpus/semeval2017-task10" or "./kleis_data/corpus/semeval2017-task10"

```
$ ls ~/kleis_data/corpus/semeval2017-task10

brat_config  eval.py       __MACOSX            README_data.md  scienceie2017_test_unlabelled  train2   xml_utils.py
dev          eval_py27.py  README_data_dev.md  README.md       semeval_articles_test          util.py  zips
```

## Test

You can test your installation with [keyphrase-extraction-example.py](https://github.com/sdhdez/kleis-keyphrase-extraction/blob/master/keyphrase-extraction-example.py)

```
$ python keyphrase-extraction-example.py
```

Also, see [here](https://github.com/sdhdez/kleis-keyphrase-extraction/blob/r0.1.2/notebooks/Keyphrase_extraction.ipynb) for another example.


## Requirements 

 - Python 3 (Tested: 3.6.5)
 - nltk (with corpus) (Tested: 3.2.5)
 - python-crfsuite (Tested: 0.9.5)
 
## Optional

### Notebooks

To run the noteooks in this repository install JupyterLab.

```
$ pip install jupyterlab
```

Then run the following command. 

```
jupyter lab
```

## Further information

This method uses a CRFs model (Conditional Random Fields) to label keyphrases in text, the model is trained with keyphrase candidates filtered with Part-of-Spech tag sequences. It is based on the method described [here](https://aclanthology.coli.uni-saarland.de/papers/S17-2174/s17-2174), but with a better performance. Please, feel free to send us comments or questions.

In this version we use [python-crfsuite](https://github.com/scrapinghub/python-crfsuite). 
