"""resources/dataset 

Module to load corpus

"""
import os, re
from pathlib import Path
from config.config import *

def get_files(path_to_files):
    for dirpath, dirnames, filenames in os.walk(path_to_files):
        for fname in filenames:
            yield dirpath + "/" + fname

def get_files_by_ext(path_to_files, ext = "txt"):
    match_ext = re.compile(r"\." + ext + "$", re.IGNORECASE)
    if len(ext) > 0:
        for fname in get_files(path_to_files):
            if match_ext.search(fname):
                yield fname

def is_path_here(path_to_corpus):
    if path_to_corpus and type(path_to_corpus) == str:
        return os.path.exists(Path(path_to_corpus)) 
    else: 
        return False

def get_dataset_paths(dataset):
    valid_paths = {}
    for name in dataset:
        if is_path_here(dataset[name]):
            valid_paths[name] = dataset[name]
    return valid_paths

def load_dataset_semeval2017task10(corpus = CORPUS_SEMEVAL2017_TASK10):
    if type(corpus) == dict and "dataset" in corpus and type(corpus["dataset"]) == dict:
        return get_dataset_paths(corpus["dataset"])
    else:
        return None
