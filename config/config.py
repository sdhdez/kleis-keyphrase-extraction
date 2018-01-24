corpus = {
        "acl-rd-tec-2.0": {
            "_id": "acl-rd-tec-2.0",
            "options": {}
            },
        "semeval2017-task10": {
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
                }
            },
            "options": {}
        }

CORPUS_DEFAULT = corpus["semeval2017-task10"]
CORPUS_SEMEVAL2017_TASK10 = corpus["semeval2017-task10"]
CORPUS_ACL_RD_TEC_2_0 = corpus["acl-rd-tec-2.0"]
