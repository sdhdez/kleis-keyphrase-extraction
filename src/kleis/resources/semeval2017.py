"""resouces/semeval2017

Class SemEval2017.

"""
from kleis.config.config import SEMEVAL2017
from kleis.resources.corpus import Corpus
import kleis.resources.dataset as kl

class SemEval2017(Corpus):
    """Class for SemEval 2017 Task 10 corpus"""
    def __init__(self):
        self._name = SEMEVAL2017
        self._lang = "en"
        super().__init__()

    def load_train(self):
        """Set train dataset"""
        dataset = dict(kl.load_dataset_raw(self._config['train-labeled'],
                                           [".txt", ".ann", ".xml"],
                                           suffix="txt",
                                           encoding=self._encoding))
        kl.preprocess_dataset(dataset, lang=self._lang)
        self._train = dataset

    def load_dev(self):
        """Set dev dataset"""
        dataset = dict(kl.load_dataset_raw(self._config['dev-labeled'],
                                           [".txt", ".ann", ".xml"],
                                           suffix="txt",
                                           encoding=self._encoding))
        kl.preprocess_dataset(dataset, lang=self._lang)
        self._dev = dataset

    def load_test(self):
        """Set test dataset"""
        dataset = dict(kl.load_dataset_raw(self._config['test-labeled'],
                                           [".txt", ".ann"],
                                           suffix="txt",
                                           encoding=self._encoding))
        dataset_part = kl.load_dataset_raw(self._config['test-unlabeled'],
                                           [".xml"],
                                           suffix="txt",
                                           encoding=self._encoding)
        for key, value in dataset_part:
            dataset[key]["raw"]["xml"] = value["raw"]["xml"]
        kl.preprocess_dataset(dataset, lang=self._lang)
        self._test = dataset
