"""resources/dataset

Module to load corpus

"""
import os
from pathlib import Path
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tag.perceptron import PerceptronTagger

from config.config import CORPUS, CORPUS_DEFAULT, SEMEVAL2017

def get_files(path_to_files):
    """Walk in path"""
    for dirpath, _, filenames in os.walk(path_to_files):
        for filename in filenames:
            yield Path(dirpath + filename)

def get_files_by_ext(path_to_files, suffix="txt"):
    """Receive path and return files found by extension"""
    if suffix:
        suffix = "." + suffix
        for filename in get_files(path_to_files):
            if filename.suffix == suffix:
                yield filename

def get_content(filename, suffixes=None, encoding="utf-8"):
    """Receive Path to file and extensions and return dict with file content"""
    suffixes = suffixes if suffixes is not None else [".txt"]
    raw = {}
    for ext in suffixes:
        tmp_filename = filename.with_suffix(ext) if filename.suffix != ext else filename
        if path_exists(tmp_filename):
            with open(str(tmp_filename), 'rt', encoding=encoding) as fin:
                raw[tmp_filename.suffix[1:]] = fin.read()
    return raw if raw else None

def path_exists(path):
    """Return true if the path exists, false otherwise"""
    it_exists = False
    if issubclass(path.__class__, str):
        it_exists = Path(path).exists()
    elif issubclass(path.__class__, Path):
        it_exists = path.exists()
    return it_exists

def get_corpus_paths(corpus):
    """Return dictionary with existing paths"""
    valid_paths = {}
    for name in corpus:
        if path_exists(corpus[name]):
            valid_paths[name] = corpus[name]
    return valid_paths

def load_config_corpus(name=None):
    """Return existing paths of the corpus"""
    corpus_paths = None
    if name is None or (isinstance(name, str) and name in CORPUS):
        corpus = CORPUS[name] if name is not None else CORPUS_DEFAULT
        if isinstance(corpus, dict) and "dataset" in corpus \
                and isinstance(corpus["dataset"], dict):
            corpus_paths = get_corpus_paths(corpus["dataset"])
    return corpus_paths

def load_corpus(name=None):
    """Return corpus object"""
    corpus = name if name else SEMEVAL2017
    obj = None
    if corpus == SEMEVAL2017:
        from resources.semeval2017 import SemEval2017
        obj = SemEval2017()
    return obj

def load_dataset_raw(dataset_name_config, extensions, suffix="txt", encoding="utf-8"):
    """Receive config dataset and teturn dataset dict"""
    if dataset_name_config:
        labeled = get_files_by_ext(dataset_name_config, suffix=suffix)
        for filename in labeled:
            yield filename.stem, {"raw": get_content(filename, extensions, encoding=encoding)}
    else:
        yield None, None

def parse_brat_content(brat_content, lang="en"):
    """Receive raw content in brat format and
    return list with parsed annotations"""
    annotations = None
    text = brat_content["txt"]
    terms = list(filter(lambda t: t[:1] == "T", brat_content["ann"].split("\n")))
    if lang == "en":
        # Tokenize
        tokens_span = TreebankWordTokenizer().span_tokenize(text)
        tokens = [text[start:end] for start, end in tokens_span]
        # Token, PoSTag, Span
        tts = [tt + (tokens_span[i],) for i, tt in enumerate(PerceptronTagger().tag(tokens))]
    annotations = tts, terms
    return annotations
