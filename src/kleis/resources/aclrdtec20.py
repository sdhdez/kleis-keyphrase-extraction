"""resouces/aclrdtec20

Class ACL-RD-TEC-2.0.

"""
import sys
from kleis.config.config import ACLRDTEC20, FORMAT_ACLXML
from kleis.resources.corpus import Corpus
import kleis.resources.dataset as kl

class AclRdTec20(Corpus):
    """Class for ACL-RD-TEC-2.0 corpus"""
    def __init__(self):
        self.name = ACLRDTEC20
        self._lang = "en"
        super().__init__()

    def load_train(self, annotator="annotator1"):
        """Set training dataset"""
        self.subname = annotator if annotator in self.option_in_cfg('annotators') else ""
        annotator1 = dict(kl.load_dataset_raw(self.option_in_cfg('train-labeled-annotator1'),
                                            [".xml"],
                                            suffix="xml",
                                            encoding=self._encoding))
        annotator2 = dict(kl.load_dataset_raw(self.option_in_cfg('train-labeled-annotator2'),
                                            [".xml"],
                                            suffix="xml",
                                            encoding=self._encoding))
        # Get 50 testing doc ids from intersection
        test_keys = self.test_keys(set(annotator1.keys()), set(annotator2.keys()))        

        if annotator == "annotator1":
            train_dataset = annotator1
            test_dataset = annotator2
        else:
            train_dataset = annotator2
            test_dataset = annotator1
        # Preprocess datasets
        kl.preprocess_dataset(train_dataset,
                              lang=self._lang,
                              dataset_format=FORMAT_ACLXML)
        kl.preprocess_dataset(test_dataset,
                              lang=self._lang,
                              dataset_format=FORMAT_ACLXML)
        # Set both datasets
        self._train = {key: value for key, value in train_dataset.items() if key not in test_keys}
        self._test = {key: value for key, value in test_dataset.items() if key in test_keys}

    def test_keys(self, keys1, keys2):
        """Return list of docids for testing"""
        keys = sorted(list(keys1 & keys2))
        # Fifty indices
        indices = [0, 3, 10, 15, 16, 18, 23, 24, 25, 35, 37, 48, 52, 55, 62, 64, 
                   66, 72, 77, 79, 80, 83, 84, 85, 90, 91, 98, 102, 103, 107, 111, 
                   113, 120, 122, 124, 126, 129, 130, 133, 136, 140, 141, 143, 145, 
                   149, 154, 156, 158, 160, 163]
        return [keys[i] for i in indices]

    def load_test(self, annotator="annotator2"):
        """Call load_train"""
        self.load_train(annotator=annotator)
