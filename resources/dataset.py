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

def tokenize_en(text):
    """Receive text string and return tokens and spans"""
    tokenizer = TreebankWordTokenizer()
    tokens = []
    tokens_span = []
    for start, end in tokenizer.span_tokenize(text):
        token = text[start:end]
        # Separate ending dot "." in token
        if len(token) > 1 and token[-1] == "." and token.count(".") == 1:
            end_resize = end - 1
            tokens.append(text[start:end_resize])
            tokens_span.append((start, end_resize))
            tokens.append(text[end_resize:end])
            tokens_span.append((end_resize, end))
        else:
            tokens.append(token)
            tokens_span.append((start, end))
    return tokens, tokens_span

def tag_text_en(tokens, tokens_span):
    """Receive tokens and spans and return tuple list with tagged tokens"""
    tagger = PerceptronTagger()
    tags = []
    for i, tagged in enumerate(tagger.tag(tokens)):
        tags.append(tagged + (tokens_span[i], []))
    return tags

# Make test
def filter_keyphrases_brat(raw_ann):
    """Receive raw content in brat format and return keyphrases"""
    filter_keyphrases = map(lambda t: t.split("\t"),
                            filter(lambda t: t[:1] == "T",
                                   raw_ann.split("\n")))
    keyphrases = {}
    for keyphrase in filter_keyphrases:
        keyphrase_key = keyphrase[0]
        # Merge annotations with ";"
        if ";" in keyphrase[1]:
            label_span = keyphrase[1].replace(';', ' ').split()
            span_str = [min(label_span[1:]), max(label_span[1:])]
        else:
            label_span = keyphrase[1].split()
            span_str = label_span[1:]
        label = label_span[0]
        span = (int(span_str[0]), int(span_str[1]))
        text = keyphrase[2]
        keyphrases[keyphrase_key] = {"keyphrase-label": label,
                                     "keyphrase-span": span,
                                     "keyphrase-text": text,
                                     "tokens-indices": []}
    return keyphrases

def parse_brat_content(brat_content, lang="en"):
    """Receive raw content in brat format and
    return list with parsed annotations"""
    if lang == "en":
        tokens, tokens_span = tokenize_en(brat_content["txt"])
        tags = tag_text_en(tokens, tokens_span)
        keyphrases = filter_keyphrases_brat(brat_content["ann"])
        for tag_i, tag in enumerate(tags):
            token_start, token_end = tag[2]
            for keyphrase_key in keyphrases:
                keyphrase_start, keyphrase_end = keyphrases[keyphrase_key]["keyphrase-span"]
                if token_start >= keyphrase_start and token_end <= keyphrase_end:
                    tag[3].append(keyphrase_key)
                    keyphrases[keyphrase_key]["tokens-indices"].append(tag_i)
    return tags, keyphrases

def preprocess_dataset(raw_dataset, lang="en"):
    """Receives raw dataset and adds pre-processed dataset"""
    for key in raw_dataset:
        tags, keyphrases = parse_brat_content(raw_dataset[key]["raw"], lang=lang)
        raw_dataset[key]["tags"] = tags
        raw_dataset[key]["keyphrases"] = keyphrases

def pos_sequence_from(keyphrase, tags):
    """Receive keyphrase dict and return PoS sequence"""
    return list(map(lambda i: tags[i][1], keyphrase["tokens-indices"]))

def load_pos_sequences(dataset):
    """Receives pre-processed dataset and return PoS sequences"""
    pos_sequences = []
    if dataset:
        for key in dataset:
            for keyphrase_id in dataset[key]["keyphrases"]:
                pos_sequences.append(pos_sequence_from(dataset[key]["keyphrases"][keyphrase_id],
                                                       dataset[key]["tags"]))
    return pos_sequences
