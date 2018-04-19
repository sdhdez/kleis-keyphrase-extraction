"""resources/corpus

Generic class to load corpus.

"""

import resources.dataset as rd

class Corpus:
    """Corpus class"""
    _lang = "en"
    _encoding = "utf-8"
    _name = None
    _config = None
    _train = None
    _dev = None
    _test = None

    def __init__(self):
        self._config = rd.load_config_corpus(name=self._name)
        self.load_train()
        self.load_dev()
        self.load_test()

    @property
    def name(self):
        """Return corpus name"""
        return self._name

    @property
    def config(self):
        """Return corpus config"""
        return self._config

    @property
    def train(self):
        """Return train dataset"""
        return self._train

    @train.deleter
    def train(self):
        """Del train dataset"""
        del self._train

    def load_train(self):
        """Placeholder to load train dataset"""
        pass

    @property
    def dev(self):
        """Return dev dataset"""
        return self._dev

    @dev.deleter
    def dev(self):
        """Del dev dataset"""
        del self._dev

    def load_dev(self):
        """Placeholder to load dev dataset"""
        pass

    @property
    def test(self):
        """Return test dataset"""
        return self._test

    @test.deleter
    def test(self):
        """Del test dataset"""
        del self._test

    def load_test(self):
        """Placeholder to load test dataset"""
        pass
