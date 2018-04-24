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
        # Tokenizing
        tokens, tokens_span = tokenize_en(brat_content["txt"])
        # Tagging
        tags = tag_text_en(tokens, tokens_span)
        # Annotated keyphrases
        keyphrases = filter_keyphrases_brat(brat_content["ann"])
        # For each tagged token
        for tag_i, tag in enumerate(tags):
            # Get token span
            token_start, token_end = tag[2]
            # For each keyphrase
            for keyphrase_key in keyphrases:
                # Get span
                keyphrase_start, keyphrase_end = keyphrases[keyphrase_key]["keyphrase-span"]
                # If token belongs to keyphrase
                if token_start >= keyphrase_start and token_end <= keyphrase_end:
                    # Add id of keyphrase to token
                    tag[3].append(keyphrase_key)
                    # Add token index to keyphrase
                    keyphrases[keyphrase_key]["tokens-indices"].append(tag_i)
    # Return tags and keyphrases
    return tags, keyphrases

def preprocess_dataset(raw_dataset, lang="en"):
    """Receives raw dataset and adds pre-processed dataset"""
    for key in raw_dataset:
        tags, keyphrases = parse_brat_content(raw_dataset[key]["raw"], lang=lang)
        # Add tagged tokens with ids of keyphrases to dataset
        raw_dataset[key]["tags"] = tags
        # Add keyphrases with indices of tokens to dataset
        raw_dataset[key]["keyphrases"] = keyphrases

def pos_sequence_from(keyphrase, tags):
    """Receive keyphrase dict and return list of tags"""
    pos_sequence = list(map(lambda i: tags[i][1], keyphrase["tokens-indices"]))
    # Special case when tokenization don't match with annotation
    if pos_sequence == []:
        tagger = PerceptronTagger()
        keyphrase_tags = tagger.tag(keyphrase['keyphrase-text'].replace("-", " ").split())
        pos_sequence = list(map(lambda t: t[1], keyphrase_tags))
    return pos_sequence

def load_pos_sequences(dataset):
    """Receive pre-processed dataset and return PoS sequences"""
    pos_sequences = {}
    if dataset:
        # FOr each document
        for key in dataset:
            # For each keyphrase in document
            for keyphrase_id in dataset[key]["keyphrases"]:
                # Extract PoS sequence from keyphrase
                pos_sequence = pos_sequence_from(dataset[key]["keyphrases"][keyphrase_id],
                                                 dataset[key]["tags"])
                # Convert list of tags to string o the PoS sequence
                pos_sequence_str = " ".join(pos_sequence)
                # Save PoS sequence as list and count of occurrences
                pos_sequences.setdefault(pos_sequence_str, {"tags": pos_sequence, "count": 0})
                # Add 1 to count
                pos_sequences[pos_sequence_str]["count"] += 1
    # Return list of PoS sequences
    return pos_sequences

def filter_candidates(element, pos_sequences, context_tokens=1):
    """Receive element from dataset, list of PoS sequences and return list of candidates"""
    candidates = []
    tags = list(map(lambda token: token[1], element["tags"]))
    # Number of tokens
    tags_len = len(tags)
    # For each PoS sequence
    for key, pos_seq in pos_sequences.items():
        # Length of sequence
        pos_seq_len = len(pos_seq["tags"])
        # Candidate length
        candidates_len = tags_len - (pos_seq_len - 1)
        # For each possible candidates
        for start in range(0, candidates_len):
            # Offset
            end = start + pos_seq_len
            # If match
            if tags[start:end] == pos_seq["tags"]:
                # Add candidate
                candidates.append({
                    "span": (start, end),
                    "context-span": (min(start, max(start - context_tokens, 0)),
                                     min(end + context_tokens, tags_len - 1)),
                    "pos_seq_id": key
                })
    return candidates

def filter_all_candidates(dataset, pos_sequences, context_tokens=1):
    """Receive dataset, list of PoS sequences and add candidates to dataset"""
    # For each document
    for key in dataset:
        # Filter candidates from document in dataset
        candidates = filter_candidates(dataset[key], pos_sequences, context_tokens=context_tokens)
        dataset[key]["candidates"] = candidates
