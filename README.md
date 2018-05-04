# Kleis: Python package for keyphrase extraction

## Install 

### Pip

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

Note: This method doesn't include pre-trained models, you should    

## Usage 

Example [here](https://github.com/sdhdez/kleis-keyphrase-extraction/blob/master/minimal-example.ipynb)


### Datasets

Thepackage already includes some pre-trained models but if you want to test by your own you should download the datasets. 

    - [SemEval 2017 Task 10](https://scienceie.github.io/resources.html) and decompress in "~/kleis_data/corpus/semeval2017-task10" or "./kleis_data/corpus/semeval2017-task10"

```
$ ls ~/kleis_data/corpus/semeval2017-task10

brat_config  eval.py       __MACOSX            README_data.md  scienceie2017_test_unlabelled  train2   xml_utils.py
dev          eval_py27.py  README_data_dev.md  README.md       semeval_articles_test          util.py  zips
```

### Test

After istalling you can test by running [keyphrase-extraction-example.py](https://github.com/sdhdez/kleis-keyphrase-extraction/blob/master/keyphrase-extraction-example.py)

```
$ python keyphrase-extraction-example.py
```

See [here](https://github.com/snovd/keyphrase-extraction/blob/master/Keyphrase_extraction.ipynb) for more deatils.


## Requirements 

 - Python 3 (Tested: 3.6.5)
 - nltk (with corpus) (Tested: 3.2.5)
 - python-crfsuite (Tested: 0.9.5)
 
## Optional

### Notebooks

To run the examples in the noteooks Keyphrase_extraction.ipynb install JupyterLab

```
$ pip install jupyterlab
```

Then run the following command. 

```
jupyter lab
```
