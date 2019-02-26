"""resouces/aclrdtec20

Class ACL-RD-TEC-2.0.

"""
from kleis.config.config import ACLRDTEC20, FORMAT_ACLXML
from kleis.resources.corpus import Corpus
import kleis.resources.dataset as kl

class AclRdTec20(Corpus):
    """Class for ACL-RD-TEC-2.0 corpus"""
    def __init__(self):
        self._name = ACLRDTEC20
        self._lang = "en"
        super().__init__()

    @property
    def subname(self):
        """Return corpus name"""
        return self._subname

    @subname.setter
    def subname(self, subname):
        """Change name"""
        self._subname = ACLRDTEC20 + "-" + subname

    def load_train(self, annotator="annotator1"):
        """Set train dataset"""
        self.subname = annotator
        dataset = dict(kl.load_dataset_raw(self._config['train-labeled-' + annotator],
                                           [".xml"],
                                           suffix="xml",
                                           encoding=self._encoding))
        kl.preprocess_dataset(dataset,
                              lang=self._lang,
                              dataset_format=FORMAT_ACLXML)
        self._train = dataset

    def load_test(self):
        """Set test dataset"""
        dataset = dict(kl.load_dataset_raw(self._config['test-labeled'],
                                           [".xml"],
                                           suffix="xml",
                                           encoding=self._encoding))
        kl.preprocess_dataset(dataset,
                              lang=self._lang,
                              dataset_format=FORMAT_ACLXML)
        self._test = dataset
