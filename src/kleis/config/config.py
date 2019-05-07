"""config/config

Default corpus configs.

"""
import sys
import inspect
from pathlib import Path

from kleis import kleis_data

ACLRDTEC20 = "acl-rd-tec-2.0"
SEMEVAL2017 = "semeval2017-task10"
FORMAT_ACLXML = "acl-xml"
FORMAT_BRAT = "brat"

KPEXTDATA_PATH = str(Path(inspect.getfile(kleis_data)).parent)

# Check for default corpus paths
DEFAULT_CORPUS_PATH = "corpus/" # + SEMEVAL2017 + "/"
if hasattr(Path, "home") and Path(str(Path.home()) + "/kleis_data/" + DEFAULT_CORPUS_PATH).exists():
    CORPUS_BASE_PATH = str(Path.home()) + "/kleis_data/" + DEFAULT_CORPUS_PATH
elif Path("./kleis_data/" + DEFAULT_CORPUS_PATH).exists():
    CORPUS_BASE_PATH = "./kleis_data/" + DEFAULT_CORPUS_PATH
elif Path(KPEXTDATA_PATH + "/" + DEFAULT_CORPUS_PATH).exists():
    CORPUS_BASE_PATH = KPEXTDATA_PATH + "/" + DEFAULT_CORPUS_PATH
else:
    print("Warning: corpus paths doesn't exists.", file=sys.stderr)
    # print("    - Download from here https://scienceie.github.io/resources.html",
    #       file=sys.stderr)
    print("    - Use one of the following paths to extract your corpus.", file=sys.stderr)
    print("        + %s/kleis_data/%s" % (str(Path.home()), DEFAULT_CORPUS_PATH), file=sys.stderr)
    print("        + ./kleis_data/%s" % DEFAULT_CORPUS_PATH, file=sys.stderr)
    print("        + %s" % (KPEXTDATA_PATH + "/" + DEFAULT_CORPUS_PATH), file=sys.stderr)
    print("    - Or you can use the pre-trained models.", file=sys.stderr)
    CORPUS_BASE_PATH = "~/kleis_data/" + DEFAULT_CORPUS_PATH
    print("Default: ", Path(CORPUS_BASE_PATH))

CORPUS_SEMEVAL2017_PATH = CORPUS_BASE_PATH + SEMEVAL2017 + "/"
CORPUS_ACLRDTEC20_PATH = CORPUS_BASE_PATH + ACLRDTEC20 + "/"

CORPUS = {"options": {}}
CORPUS[ACLRDTEC20] = {
            "_id": ACLRDTEC20,
            "format": "xml",
            "format-description": "XML",
            "url": "https://github.com/languagerecipes/acl-rd-tec-2.0/",
            "dataset": {
                "annotators": set(["annotator1", "annotator2"]),
                "train-labeled-annotator1": CORPUS_ACLRDTEC20_PATH + \
                    "distribution/annoitation_files/annotator1/",
                "train-labeled-annotator2": CORPUS_ACLRDTEC20_PATH + \
                    "distribution/annoitation_files/annotator2/",
                "train-unlabeled": None,
                "dev-labeled": None,
                "dev-unlabeled": None,
                "test-unlabeled": None,
                "test-labeled": None
                },
            "options": {}
        }
CORPUS[SEMEVAL2017] = {
            "_id": SEMEVAL2017,
            "format": "brat",
            "format-description": "brat standoff format, http://brat.nlplab.org/standoff.html",
            "dataset": {
                "train-labeled": CORPUS_SEMEVAL2017_PATH + "train2/",
                "train-unlabeled": None,
                "dev-labeled": CORPUS_SEMEVAL2017_PATH + "dev/",
                "dev-unlabeled": None,
                "test-unlabeled": CORPUS_SEMEVAL2017_PATH + "scienceie2017_test_unlabelled/",
                "test-labeled": CORPUS_SEMEVAL2017_PATH + "semeval_articles_test/"
                },
            "options": {}
        }

CORPUS_DEFAULT = CORPUS[SEMEVAL2017]
CORPUS_SEMEVAL2017_TASK10 = CORPUS[SEMEVAL2017]
CORPUS_ACL_RD_TEC_2_0 = CORPUS[ACLRDTEC20]

# Check for default paths for models
DEFAULT_MODELS_PATH = "models/"
if hasattr(Path, "home") and Path(str(Path.home()) + "/kleis_data/" + DEFAULT_MODELS_PATH).exists():
    MODELS_PATH = str(Path.home()) + "/kleis_data/" + DEFAULT_MODELS_PATH
elif Path("./kleis_data/" + DEFAULT_MODELS_PATH).exists():
    MODELS_PATH = "./kleis_data/" + DEFAULT_MODELS_PATH
elif Path(KPEXTDATA_PATH + "/" + DEFAULT_MODELS_PATH).exists():
    MODELS_PATH = KPEXTDATA_PATH + "/" + DEFAULT_MODELS_PATH
else:
    print("Warning: Path to save models doesn't exists.", file=sys.stderr)
    print("    - Possible paths are:", file=sys.stderr)
    print("        + %s" % (str(Path.home()) + "/kleis_data/" + DEFAULT_MODELS_PATH), file=sys.stderr)
    print("        + %s" % ("./kleis_data/" + DEFAULT_MODELS_PATH), file=sys.stderr)
    print("        + %s" % (KPEXTDATA_PATH + "/" + DEFAULT_MODELS_PATH), file=sys.stderr)
    print("    - Default will be %s" % DEFAULT_MODELS_PATH, file=sys.stderr)
    MODELS_PATH = DEFAULT_MODELS_PATH

# Check for default paths for PoS tag sequences
DEFAULT_TRAIN_PATH = "train/"
if Path("./kleis_data/" + DEFAULT_TRAIN_PATH).exists():
    TRAIN_PATH = "./kleis_data/" + DEFAULT_TRAIN_PATH
elif Path(str(Path.home()) + "/kleis_data/" + DEFAULT_TRAIN_PATH).exists():
    TRAIN_PATH = str(Path.home()) + "/kleis_data/" + DEFAULT_TRAIN_PATH
elif Path(KPEXTDATA_PATH + "/" + DEFAULT_TRAIN_PATH).exists():
    TRAIN_PATH = KPEXTDATA_PATH + "/" + DEFAULT_TRAIN_PATH
else:
    print("Warning: Path to save train models doesn't exists.", file=sys.stderr)
    print("    - Possible paths are:", file=sys.stderr)
    print("        + %s" % ("./" + DEFAULT_TRAIN_PATH), file=sys.stderr)
    print("        + %s" % (str(Path.home()) + "/kleis_data/" + DEFAULT_TRAIN_PATH), file=sys.stderr)
    print("        + %s" % (KPEXTDATA_PATH + "/" + DEFAULT_TRAIN_PATH), file=sys.stderr)
    print("    - Default will be %s" % DEFAULT_TRAIN_PATH, file=sys.stderr)
    TRAIN_PATH = DEFAULT_TRAIN_PATH

OUTPUT_PATH = "output/"
