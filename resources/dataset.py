"""resources/dataset

Module to load corpus

"""
import os
import re
from pathlib import Path
from config import config as cfg

def get_files(path_to_files):
    """Walk in path"""
    for dirpath, _, filenames in os.walk(path_to_files):
        for fname in filenames:
            yield dirpath + "/" + fname

def get_files_by_ext(path_to_files, ext="txt"):
    """Return files by extension"""
    match_ext = re.compile(r"\." + ext + "$", re.IGNORECASE)
    if ext:
        for fname in get_files(path_to_files):
            if match_ext.search(fname):
                yield fname

def path_exists(path_to_corpus):
    """Return true if the path exists, false otherwise"""
    it_exists = False
    if path_to_corpus and isinstance(path_to_corpus, str):
        it_exists = os.path.exists(Path(path_to_corpus))
    return it_exists

def get_dataset_paths(dataset):
    """Return dictionary with existing paths"""
    valid_paths = {}
    for name in dataset:
        if path_exists(dataset[name]):
            valid_paths[name] = dataset[name]
    return valid_paths

def load_dataset_semeval2017task10(corpus=None):
    """Return existing paths of the dataset"""
    corpus = corpus if corpus is not None else cfg.CORPUS_SEMEVAL2017_TASK10
    dataset_paths = None
    if isinstance(corpus, dict) and "dataset" in corpus \
            and isinstance(corpus["dataset"], dict):
        dataset_paths = get_dataset_paths(corpus["dataset"])
    return dataset_paths
