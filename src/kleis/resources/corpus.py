"""resources/corpus

Generic class to load corpus.

"""
import sys
import copy
import kleis.resources.dataset as kl
from kleis.methods import crf as kcrf

class Corpus:
    """Corpus class"""
    _lang = "en"
    _encoding = "utf-8"
    _name = None
    _subname = None
    _config = None
    _train = None
    _dev = None
    _test = None
    _pos_sequences = None
    _annotated_candidates_spans = None
    _annotated_candidates = None
    _generic_label = True
    _features_method = None
    _tagging_notation = None
    _context_tokens = None
    _crf_tagger = None
    _crf_method = None
    _filter_min_count = 3

    def __init__(self):
        self.config = kl.load_config_corpus(name=self.name)
        # self.load_train()
        # self.load_dev()
        # self.load_test()
        # self.training()
        # self.load_pos_sequences()

    def load_pos_sequences(self, filter_min_count=1):
        """Load PoS sequences"""
        self._filter_min_count = filter_min_count
        # Load pos sequences
        self.pos_sequences = self.train

    def training(self, features_method="simple", tagging_notation="BILOU",
                 context_tokens=1, crf="pycrfsuite", filter_min_count=3,
                 generic_label=True):
        """Init training"""
        self._features_method = features_method
        self.load_pos_sequences(filter_min_count=filter_min_count)
        self.annotated_candidates_spans = self.train
        # CRF train
        self.crf_train(features_method=features_method,
                       tagging_notation=tagging_notation,
                       context_tokens=context_tokens,
                       generic_label=generic_label, crf=crf)

    @property
    def name(self):
        """Return corpus name"""
        return self._name

    @name.setter
    def name(self, name):
        """Set corpus name"""
        self._name = name
    
    @name.deleter
    def name(self):
        """Delete corpus name"""
        del self._name

    @property
    def subname(self):
        """Return subname"""
        return self._subname if self._subname else ""
    
    @subname.setter
    def subname(self, subname):
        """Set subname"""
        self._subname = subname

    @subname.deleter 
    def subname(self):
        """Delete subname"""
        del self._subname

    @property
    def fullname(self):
        """Return fullname"""
        return self.name + ("-" + self.subname if self.subname else "")

    @property
    def config(self):
        """Return corpus config"""
        return self._config

    @config.setter
    def config(self, config):
        """Return corpus config"""
        self._config = config

    def option_in_cfg(self, option):
        """Get option from config"""
        if option in self.config:
            return self.config[option]
        else:
            print("Option config[%s] doesn't exists (%s)." % (option, self.name), file=sys.stderr)
            return None

    @property
    def train(self):
        """Return train dataset"""
        return {k:v for k, v in list(self._train.items())} if self._train else None

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
        """Return pos sequences"""
        return {str(key): value \
                for key, value in filter(lambda ps: ps[1]["count"] >= self._filter_min_count,
                                         self._pos_sequences.items())}

    @pos_sequences.setter
    def pos_sequences(self, dataset):
        """Load PoS sequences from dataset, using format
        returned by kl.parse_brat_content()"""
        self._pos_sequences = kl.load_pos_sequences(dataset, name=self.fullname)

    @pos_sequences.deleter
    def pos_sequences(self):
        """Placeholder to load pos sequences"""
        del self._pos_sequences

    def filtering_counts(self):
        """Return filtering counts"""
        if self._pos_sequences is None:
            print("Warning: There aren't PoS tag sequences loaded. Use self.load_pos_sequences().", file=sys.stderr)
            return []
        return sorted(list({posseqs['count'] for posseqs in self._pos_sequences.values()}))

    @property
    def annotated_candidates_spans(self):
        """Extract annotated candidate phrases from train dataset"""
        return copy.deepcopy(self._annotated_candidates_spans)

    @annotated_candidates_spans.setter
    def annotated_candidates_spans(self, dataset):
        """Set annotated candidates_spans"""
        self._annotated_candidates_spans = kl.filter_all_candidates_spans(dataset,
                                                                          self.pos_sequences,
                                                                          annotated=True)

    @annotated_candidates_spans.deleter
    def annotated_candidates_spans(self):
        """Delete annotated candidates_spans"""
        del self._annotated_candidates_spans

    @property
    def annotated_candidates(self):
        """Get training features for dataset"""
        return copy.deepcopy(self._annotated_candidates)

    @annotated_candidates.setter
    def annotated_candidates(self, candidates_spans):
        """Set training features"""
        self._annotated_candidates = kl.dataset_features_labels_from(
            candidates_spans,
            self.train,
            context_tokens=self._context_tokens,
            features_method=self._features_method,
            tagging_notation=self._tagging_notation)

    @annotated_candidates.deleter
    def annotated_candidates(self):
        """Set training features"""
        del self._annotated_candidates

    def crf_train(self, features_method="simple",
                  tagging_notation="BILOU", context_tokens=1,
                  generic_label=True, crf="pycrfsuite"):
        """Training CRF"""
        self._crf_method = crf
        self._features_method = features_method
        self._tagging_notation = tagging_notation
        self._context_tokens = context_tokens
        self._generic_label = generic_label
        self.annotated_candidates = self.annotated_candidates_spans
        model_file_name = "%s.%s.%s.%s.ctx%s.lbl%s.%s" % \
            (self.fullname,
             self._filter_min_count,
             self._features_method,
             self._tagging_notation.lower(),
             self._context_tokens,
             ("generic" if self._generic_label else "annotation"),
             self._crf_method)
        if self._crf_method == "pycrfsuite":
            self._crf_tagger = kcrf.pycrfsuite_train(self.annotated_candidates,
                                                     name=model_file_name)

    @property
    def crf_tagger(self):
        """Return tagger"""
        return self._crf_tagger

    @crf_tagger.deleter
    def crf_tagger(self):
        """Delete tagger"""
        del self._crf_tagger

    def label_text(self, text, post_processing=True):
        """Labeling method"""
        keyphrases = []
        if self._crf_method == "pycrfsuite":
            keyphrases = kcrf.pycrfsuite_label(self.crf_tagger,
                                               self.pos_sequences,
                                               text,
                                               tagging_notation=self._tagging_notation)
        if post_processing:
            keyphrases = kl.post_processing(keyphrases)
        return keyphrases

    def eval(self, dataset=None, method="micro-average", beta=1.0, post_processing=True):
        """Evaluate a model using micro-average on the testing dataset"""
        # Set dataset
        if dataset is None:
            if self.test is not None:
                dataset = self.test
        # Save spans from extracted and annotated keyphrases
        spans = []
        # Extract keyphrases and evaluate
        for key in dataset:
            selected_elements = {kp[1][1] for kp in self.label_text(dataset[key]["raw"]["txt"], 
                                                                    post_processing=post_processing)}
            relevant_elements = {kp['keyphrase-span'] for kp in dataset[key]["keyphrases"].values()}
            # List of results
            spans.append((selected_elements, relevant_elements))
        # If micro-average
        if method == "micro-average":
            # Evaluate
            precision, recall, f1 = kl.microaverage(spans, beta=beta)
        return precision, recall, f1