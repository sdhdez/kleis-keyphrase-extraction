# Keyphrase extraction

## How to use 

Example [here](https://github.com/snovd/keyphrase-extraction/blob/master/Keyphrase_extraction.ipynb)

## Install 

### Clone repository.

```
$ git clone git@github.com:snovd/keyphrase-extraction.git
```

### Datasets
Download dataset files [SemEval 2017 Task 10](https://scienceie.github.io/resources.html) and decompress in "corpus/semeval2017-task10/"

```
$ ls corpus/semeval2017-task10/

brat_config  eval.py       __MACOSX            README_data.md  scienceie2017_test_unlabelled  train2   xml_utils.py
dev          eval_py27.py  README_data_dev.md  README.md       semeval_articles_test          util.py  zips
```
### Config dataset

If needed, chage paths in [config/config.py](https://github.com/snovd/keyphrase-extraction/blob/master/config/config.py) 

Default config is:

```
    SEMEVAL2017: {
        "_id": "semeval2017-task10",
        "format": "brat",
        "format-description": "brat standoff format, http://brat.nlplab.org/standoff.html",
        "dataset": {
            "train-labeled": "corpus/semeval2017-task10/train2/",
            "train-unlabeled": None,
            "dev-labeled": "corpus/semeval2017-task10/dev/",
            "dev-unlabeled": None,
            "test-unlabeled": "corpus/semeval2017-task10/scienceie2017_test_unlabelled/",
            "test-labeled": "corpus/semeval2017-task10/semeval_articles_test/"
            },
        "options": {}
        }
```
### Test

```
$ python main.py
```

See [here](https://github.com/snovd/keyphrase-extraction/blob/master/Keyphrase_extraction.ipynb) for more deatils.

## Requirements 

 - Python 3 (Tested: 3.6.5)
 - nltk (with corpus) (Tested: 3.2.5)
 - python-crfsuite (Tested: 0.9.5)
 
To install requirements.

```
pip install nltk python-crfsuite
```

## Optional

### Notebooks

To run the example in the noteooks Keyphrase_extraction.ipynb install JupyterLab

```
$ pip install jupyterlab
```

Then run the following command. 

```
jupyter lab
```

#### Docker and JupyterLab

You can run the example with this [docker image](https://hub.docker.com/r/sdavidhdez/keyphrase-extraction/) or you ca build it yourself with the Dockerfile.

To run the pre-builded image.

```
sudo docker run --rm -p '8888:8888' -v '/host/path/to/the/package/keyphrase-extraction/:/home/someuser/notebooks/keyphrase-extraction' -v '/host/path/to/nltk_data:/home/someuser/nltk_data' sdavidhdez/keyphrase-extraction:latest
```

Copy the displayed token from the command-line, go to http://localhost:8888 and paste the token. 

Change the volumes paths (/host/path/to/the/package/keyphrase-extraction/, /host/path/to/nltk_data:/home/someuser/nltk_data) to whatever you need.



