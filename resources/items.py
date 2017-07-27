"""resources/items 

Module to load corpus

"""
import os, re
from config.config import *

def get_files(path_to_files):
    for dirpath, dirnames, filenames in os.walk(path_to_files):
        for fname in filenames:
            yield dirpath + " " + fname

def get_files_by_ext(path_to_files, ext = "txt"):
    match_ext = re.compile(r"\." + ext + "$", re.IGNORECASE)
    if len(ext) > 0:
        for fname in get_files(path_to_files):
            if match_ext.search(fname):
                yield fname

def load_corpus_files(corpus = CORPUS_DEFAULT):
    return corpus
