"""config/config

Default corpus configs.

"""
import os
from pathlib import Path

ACLRDTEC = "acl-rd-tec-2.0"
SEMEVAL2017 = "semeval2017-task10"

# Check for default paths for corpus
DEFAULT_CORPUS_PATH = "corpus/"
if Path(DEFAULT_CORPUS_PATH).exists():
    CORPUS_PATH = DEFAULT_CORPUS_PATH
elif Path("./" + DEFAULT_CORPUS_PATH).exists():
    CORPUS_PATH = "./" + DEFAULT_CORPUS_PATH
elif Path("~/" + DEFAULT_CORPUS_PATH).exists():
    CORPUS_PATH = "~/" + DEFAULT_CORPUS_PATH
else:
    os.mkdir(DEFAULT_CORPUS_PATH)
    CORPUS_PATH = DEFAULT_CORPUS_PATH

CORPUS = {
    ACLRDTEC: {
        "_id": "acl-rd-tec-2.0",
        "options": {}
        },
    SEMEVAL2017: {
        "_id": "semeval2017-task10",
        "format": "brat",
        "format-description": "brat standoff format, http://brat.nlplab.org/standoff.html",
        "dataset": {
            "train-labeled": CORPUS_PATH + SEMEVAL2017 + "/train2/",
            "train-unlabeled": None,
            "dev-labeled": CORPUS_PATH + SEMEVAL2017 + "/dev/",
            "dev-unlabeled": None,
            "test-unlabeled": CORPUS_PATH + SEMEVAL2017 + "/scienceie2017_test_unlabelled/",
            "test-labeled": CORPUS_PATH + SEMEVAL2017 + "/semeval_articles_test/"
            },
        "options": {}
        },
    "options": {}
}
CORPUS_DEFAULT = CORPUS[SEMEVAL2017]
CORPUS_SEMEVAL2017_TASK10 = CORPUS[SEMEVAL2017]
CORPUS_ACL_RD_TEC_2_0 = CORPUS[ACLRDTEC]

# Check for default paths for models
DEFAULT_MODELS_PATH = "models/"
if Path(DEFAULT_MODELS_PATH).exists():
    MODELS_PATH = DEFAULT_MODELS_PATH
elif Path("./" + DEFAULT_MODELS_PATH).exists():
    MODELS_PATH = "./" + DEFAULT_MODELS_PATH
elif Path("~/keyphrase-extraction/crf-" + DEFAULT_MODELS_PATH).exists():
    MODELS_PATH = "~/keyphrase-extraction/crf-" + DEFAULT_MODELS_PATH
elif Path("~/crf-" + DEFAULT_MODELS_PATH).exists():
    MODELS_PATH = "~/crf-" + DEFAULT_MODELS_PATH
elif Path("~/" + DEFAULT_MODELS_PATH).exists():
    MODELS_PATH = "~/" + DEFAULT_MODELS_PATH
else:
    os.mkdir(DEFAULT_MODELS_PATH)
    MODELS_PATH = DEFAULT_MODELS_PATH

OUTPUT_PATH = "output/"
