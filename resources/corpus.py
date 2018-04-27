"""resources/corpus

Generic class to load corpus.

"""
import resources.dataset as rd
import methods.crf as mc
import copy

class Corpus:
    """Corpus class"""
    _lang = "en"
    _encoding = "utf-8"
    _name = None
    _config = None
    _train = None
    _dev = None
    _test = None
    _pos_sequences = None
    _annotated_candidates_spans = None
    _annotated_candidates = None
    _method_features = None
    _context_tokens = None
    _crf_tagger = None
    _crf_method = None
    _filter_min_count = 1

    def __init__(self):
        self._config = rd.load_config_corpus(name=self._name)
        self.load_train()
        self.load_dev()
        self.load_test()
        # self.training()
        self.load_pos_sequences()

    def load_pos_sequences(self, filter_min_count=1):
        """Load PoS sequences"""
        self._filter_min_count = filter_min_count
        # Load pos sequences
        self.pos_sequences = self._train

    def training(self, filter_min_count=1):
        """Init training"""
        self.load_pos_sequences(filter_min_count=filter_min_count)
        self.annotated_candidates_spans = self._train
        # CRF train
        self.crf_train()

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
        return copy.deepcopy(self._train)

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
        return copy.deepcopy(self._dev)

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
        return copy.deepcopy(self._test)

    @test.deleter
    def test(self):
        """Del test dataset"""
        del self._test

    def load_test(self):
        """Placeholder to load test dataset"""
        pass

    @property
    def pos_sequences(self):
        """Placeholder to load pos sequences"""
        return {str(key): value \
                for key, value in filter(lambda ps: ps[1]["count"] >= self._filter_min_count,
                                         self._pos_sequences.items())}


    @pos_sequences.setter
    def pos_sequences(self, dataset):
        """Load PoS sequences from dataset, using format
        returned by rd.parse_brat_content()"""
        if dataset:
            self._pos_sequences = rd.load_pos_sequences(dataset)
        else:
            self._pos_sequences = None

    @pos_sequences.deleter
    def pos_sequences(self):
        """Placeholder to load pos sequences"""
        del self._pos_sequences

    @property
    def annotated_candidates_spans(self):
        """Extract annotated candidate phrases from train dataset"""
        return self._annotated_candidates_spans

    @annotated_candidates_spans.setter
    def annotated_candidates_spans(self, dataset):
        """Set annotated candidates_spans"""
        self._annotated_candidates_spans = rd.filter_all_candidates_spans(dataset,
                                                                          self.pos_sequences,
                                                                          annotated=True)

    @annotated_candidates_spans.deleter
    def annotated_candidates_spans(self):
        """Delete annotated candidates_spans"""
        del self._annotated_candidates_spans

    @property
    def annotated_candidates(self):
        """Get training features for dataset"""
        return self._annotated_candidates

    @annotated_candidates.setter
    def annotated_candidates(self, candidates_spans):
        """Set training features"""
        self._annotated_candidates = rd.dataset_features_labels_from(
            candidates_spans,
            self._train,
            context_tokens=self._context_tokens,
            method=self._method_features)

    @annotated_candidates.deleter
    def annotated_candidates(self):
        """Set training features"""
        del self._annotated_candidates

    def crf_train(self, method="simple", context_tokens=1, crf="pycrfsuite"):
        """Training CRF"""
        self._crf_method = crf
        self._method_features = method
        self._context_tokens = context_tokens
        self.annotated_candidates = self._annotated_candidates_spans
        if self._crf_method == "pycrfsuite":
            self._crf_tagger = mc.pycrfsuite_train(self.annotated_candidates,
                                                   name="%s.%s.%s.%s" % ("candidates-model",
                                                                         self._filter_min_count,
                                                                         self._method_features,
                                                                         self._crf_method))

    @property
    def crf_tagger(self):
        """Return tagger"""
        return self._crf_tagger

    @crf_tagger.deleter
    def crf_tagger(self):
        """Delete tagger"""
        del self._crf_tagger

    def label_text(self, text):
        """Labeling method"""
        keyphrase = []
        if self._crf_method == "pycrfsuite":
            keyphrase = mc.pycrfsuite_label(self._crf_tagger, self.pos_sequences, text)
        return keyphrase
