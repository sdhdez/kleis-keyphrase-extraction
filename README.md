# Keyphrase extraction

Example [here](https://github.com/snovd/keyphrase-extraction/blob/master/Keyphrase_extraction.ipynb)

Clone repository.

```
git clone git@github.com:snovd/keyphrase-extraction.git
```

Download dataset files [SemEval 2017 Task 10](https://scienceie.github.io/resources.html) and decompress in "corpus/semeval2017-task10/"

```
$ ls corpus/semeval2017-task10/

brat_config  eval.py       __MACOSX            README_data.md  scienceie2017_test_unlabelled  train2   xml_utils.py
dev          eval_py27.py  README_data_dev.md  README.md       semeval_articles_test          util.py  zips
```

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
