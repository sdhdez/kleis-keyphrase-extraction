"""resources/dataset

Module to load corpus

"""
import os
import sys
from pathlib import Path
import pickle
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tag.perceptron import PerceptronTagger
import xml.etree.ElementTree as ET
import re

from kleis.config.config import CORPUS, CORPUS_DEFAULT,\
                                SEMEVAL2017, ACLRDTEC20,\
                                TRAIN_PATH, FORMAT_ACLXML, FORMAT_BRAT

def get_files(path_to_files):
    """Walk in path"""
    for dirpath, _, filenames in os.walk(path_to_files):
        for filename in filenames:
            yield Path(os.path.join(dirpath, filename))

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

def load_config_corpus(name=None):
    """Return existing paths of the corpus"""
    corpus_config = None
    # If name is str and is defined in CORPUS
    corpus = CORPUS[name] if (isinstance(name, str) \
        and name in CORPUS) else CORPUS_DEFAULT
    # If corpus is dict and has a dataset 
    if isinstance(corpus, dict) and "dataset" in corpus \
        and isinstance(corpus["dataset"], dict):
        corpus_config = corpus["dataset"]
    return corpus_config

def load_corpus(name=None):
    """Return corpus object"""
    corpus = name if name else SEMEVAL2017
    obj = None
    if corpus == SEMEVAL2017:
        from kleis.resources.semeval2017 import SemEval2017
        obj = SemEval2017()
    if corpus == ACLRDTEC20:
        from kleis.resources.aclrdtec20 import AclRdTec20
        obj = AclRdTec20()
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

def filter_keyphrases_acl_xml(root_xml_content, text_nodes, exclude_title=True):
    """Receive raw content in XML ACL-RD-TEC 2.0 format and return keyphrases"""
    # Dict with keyphrases info
    keyphrases = {}
    # List of strings (keyphrases)
    keyphrases_list = []
    # Used to give consistency on the ACL-RD-TEC corpus.
    if exclude_title:
        # Get first section element to exclude annotations on titles
        element = root_xml_content.find("Section")
    else:
        # Include title annotations
        element = root_xml_content
    # Iter over annotated terms
    for i, term in enumerate(element.iter("term"), 1):
        # Add annotated keyphrase to dict
        keyphrases["T" + str(i)] = {"keyphrase-text": term.text,
                                    "keyphrase-label": term.get("class"),
                                    # There is not span
                                    "keyphrase-span": None,
                                    "tokens-indices": []}
        # Add text from term to list of keyphrases
        keyphrases_list.append(term.text)
    # Number of keyphrases found
    kp_found = 0
    # Content of the document
    text_content = ""
    # Iter over sentence nodes from XML 
    for sentence in text_nodes:
        # Iter over nodes in sentence
        for chunk in sentence:
            # Add text node (chunk) to text content
            text_content += chunk
            # If text node match the first keyphrase in list
            if keyphrases_list and keyphrases_list[0] == chunk:
                # One keyphrase found
                kp_found += 1
                # Set end span
                span_end = len(text_content)
                # Set start span
                span_start = span_end - len(chunk)
                # Add span to corresponding keyphrase
                keyphrases["T" + str(kp_found)]["keyphrase-span"] = (span_start, span_end)
                # Removes from list the found keyphrase
                keyphrases_list.pop(0)
        # Add space between sentences
        text_content += " "
    if keyphrases_list:
        print("Info: Keyphrases didn't match. Check your XML file.", keyphrases_list, file=sys.stderr)
    # Return keyphrases and text content (removing last space)
    return keyphrases, text_content if not text_content else  text_content[:-1]

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

def normalize_acl_xml(text_xml):
    """Receive a XML string and strip spaces"""
    stripped_text_xml = ""
    # If there is not a sentence marked with <S></S>
    if text_xml.find("<S>") < 0:
        # Special case for ACL-RD-TEC 2.0
        text_xml = re.sub(r'</SectionTitle>(\t+)*', '</SectionTitle>\t<S>', text_xml)
        text_xml = re.sub(r'(\s*)</Section>', r'</S>\1</Section>', text_xml)
    # Iter over xml text spliting by Tab character
    for line in text_xml.split("\t"):
        # Remove spaces (malformed XML)
        stripped_line = line.strip()
        # If not empty
        if stripped_line:
            # Remove spaces betweent tags (malformed XML)
            # Special case for ACL-RD-TEC 2.0
            stripped_text_xml += stripped_line.replace("<S>         <term", "<S><term")
    return stripped_text_xml

def text_nodes_acl_xml(root_xml_content):
    """Receive XML object and return list of lists of strings"""
    text_nodes = []
    # Iter over sentences
    for sentence in root_xml_content.iter("S"):
        # Get text nodes from sentence
        sentence_text = list(sentence.itertext())
        # Add text nodes to list
        text_nodes.append(sentence_text)
    return text_nodes

def parse_acl_xml_content(xml_content, lang="en"):
    """Receive raw content in XML ACL-RD-TEC-2.0 format and
    return list with parsed annotations"""
    root_xml_content = ET.fromstring(normalize_acl_xml(xml_content["xml"]))
    text_nodes = text_nodes_acl_xml(root_xml_content)
    # Annotated keyphrases
    keyphrases, text_content = filter_keyphrases_acl_xml(root_xml_content, text_nodes)
    if lang == "en":
        # Tokenizing
        tokens, tokens_span = tokenize_en(text_content)
        # Tagging
        tags = tag_text_en(tokens, tokens_span)
        # For each tagged token (merge with similar code above)
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
    # return tags, keyphrases and text content (XML)
    return tags, keyphrases, text_content

def preprocess_dataset_brat(raw_dataset, lang="en"):
    """Receives raw dataset and adds pre-processed dataset in brat format"""
    for key in raw_dataset:
        tags, keyphrases = parse_brat_content(raw_dataset[key]["raw"], lang=lang)
        # Add tagged tokens with ids of keyphrases to dataset
        raw_dataset[key]["tags"] = tags
        # Add keyphrases with indices of tokens to dataset
        raw_dataset[key]["keyphrases"] = keyphrases

def preprocess_dataset_acl_xml(raw_dataset, lang="en"):
    """Receives raw dataset and adds pre-processed dataset in XML format 
    used in the ACL-RD-TEC-2.0"""
    for key in raw_dataset:
        # tags, keyphrases, text_content (needed for ACL-RRD-TEC)
        tags, keyphrases, text_content = parse_acl_xml_content(raw_dataset[key]["raw"], lang=lang)
        # Add tagged tokens with ids of keyphrases to dataset
        raw_dataset[key]["tags"] = tags
        # Add keyphrases with indices of tokens to dataset
        raw_dataset[key]["keyphrases"] = keyphrases
        # Add text content to the dataset
        raw_dataset[key]["raw"]["txt"] = text_content

def preprocess_dataset(raw_dataset, lang="en", dataset_format="brat"):
    """Receives raw dataset and adds pre-processed dataset"""
    if dataset_format == FORMAT_BRAT:
        preprocess_dataset_brat(raw_dataset, lang=lang)
    if dataset_format == FORMAT_ACLXML:
        preprocess_dataset_acl_xml(raw_dataset, lang=lang)

def pos_sequence_from(keyphrase, tags):
    """Receive keyphrase dict and return list of tags"""
    pos_sequence = list(map(lambda i: tags[i][1], keyphrase["tokens-indices"]))
    # Special case when tokenization don't match with annotation
    if pos_sequence == []:
        tagger = PerceptronTagger()
        keyphrase_tags = tagger.tag(keyphrase['keyphrase-text'].split())
        pos_sequence = list(map(lambda t: t[1], keyphrase_tags))
    return pos_sequence

def load_pos_sequences(dataset, name=""):
    """Receive pre-processed dataset and return PoS sequences"""
    pos_sequences = {}
    pos_seq_path = TRAIN_PATH + "/pos-sequences." + str(name) + ".normal.pkl"
    if Path(pos_seq_path).exists():
        with open(pos_seq_path, "rb") as fin:
            pos_sequences = pickle.load(fin)
    elif dataset:
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
        with open(pos_seq_path, "wb") as fout:
            pickle.dump(pos_sequences, fout)
    # Return list of PoS sequences
    return pos_sequences

def filter_pos_sequences(element, pos_sequences, annotated=False):
    """Receive element from dataset, list of PoS sequences and return list of candidates_spans"""
    candidates_spans = []
    tags = tags_from(element)
    # For each PoS sequence
    for key, pos_seq in pos_sequences.items():
        # Length of sequence
        pos_seq_len = len(pos_seq["tags"])
        # Candidate length
        candidates_spans_len = len(tags) - (pos_seq_len - 1)
        # For each possible candidates_spans
        for start in range(0, candidates_spans_len):
            # Offset
            end = start + pos_seq_len
            # If match
            if tags[start:end] == pos_seq["tags"]:
                candidate_span = {
                    "span": (start, end),
                    "pos-seq-id": key
                }
                if annotated:
                    candidate_span_keyphrase_id = match_candidate_span_keyphrase(
                        (start, end),
                        element["tags"][start:end],
                        element["keyphrases"]
                    )
                    candidate_span["keyphrase-id"] = candidate_span_keyphrase_id
                # Add candidate_span
                candidates_spans.append(candidate_span)
    return candidates_spans

def filter_all_candidates_spans(dataset, pos_sequences, annotated=False,
                                filtering="pos-sequences"):
    """Receive dataset, list of PoS sequences and add candidates_spans to dataset"""
    dataset_candidates_spans = {}
    if dataset:
        # For each document
        for key in dataset:
            # Filter candidates_spans from document in dataset
            if filtering == "pos-sequences":
                dataset_candidates_spans[key] = filter_pos_sequences(dataset[key],
                                                                     pos_sequences,
                                                                     annotated=annotated)
            else:
                break
    return dataset_candidates_spans

def tags_from(element):
    """Receive element from dataset and return separated tags"""
    tags = list(map(lambda token: token[1], element["tags"]))
    return tags

def match_candidate_span_keyphrase(segment_span, candidate_span_segment, keyphrases):
    """Receive element from dataset and return keyphrases ids with token indices"""
    start, end = segment_span
    token_keyphrases_ids = map(lambda token: token[3], candidate_span_segment)
    keyphrases_ids = set(kpid for kps_ids in token_keyphrases_ids for kpid in kps_ids)
    candidate_span_keyphrase_id = None
    for kpid in keyphrases_ids:
        keyphrase_start = min(keyphrases[kpid]["tokens-indices"])
        keyphrase_end = max(keyphrases[kpid]["tokens-indices"]) + 1
        if start == keyphrase_start and end == keyphrase_end:
            candidate_span_keyphrase_id = kpid
            break
    return candidate_span_keyphrase_id

def dataset_features_labels_from(candidates_spans, dataset,
                                 context_tokens=1, features_method="simple",
                                 tagging_notation="BILOU", generic_label=True):
    """Receive candidates_spans and dataset and return all features from candidates"""
    dataset_features_labels = {}
    if dataset:
        dataset_features_labels = {
            key: candidates_spans_features_labels_from(
                candidates_spans[key],
                element,
                context_tokens=context_tokens,
                features_method=features_method,
                tagging_notation=tagging_notation,
                generic_label=generic_label) \
                for key, element in dataset.items()
        }
    return dataset_features_labels

def candidates_spans_features_labels_from(candidates_spans, element,
                                          context_tokens=1, features_method="simple",
                                          tagging_notation="BILOU", generic_label=True):
    """Receive candidate_span list and return list of candidate features"""
    features_labels = [
        features_labels_from(candidate_span,
                             element,
                             context_tokens=context_tokens,
                             features_method=features_method,
                             tagging_notation=tagging_notation,
                             generic_label=generic_label) \
                             for candidate_span in candidates_spans
    ]
    return features_labels

def features_labels_from(candidate_span, element, context_tokens=1,
                         features_method="simple", tagging_notation="BILOU", generic_label=True):
    """"Return candidate_span features"""
    start, end = candidate_span["span"]
    label = keyphrase_label_from(candidate_span, element, generic_label=generic_label)
    context_start = min(start, max(start - context_tokens, 0))
    context_end = end + context_tokens
    features_labels = []
    tags_segment = element["tags"][context_start:context_end]
    labels = add_notation(tags_segment, label,
                          start - context_start,
                          min(context_end, len(element["tags"])) - end,
                          tagging_notation=tagging_notation)
    if features_method == "simple":
        # Default features
        for offset, tag in enumerate(tags_segment):
            token_bc_start = max(offset - context_tokens, 0)
            features = simple_features(tag,
                                       context_start + offset,
                                       len(element["tags"]),
                                       tags_segment[token_bc_start:offset],
                                       tags_segment[offset+1:offset + context_tokens + 1]
                                      )
            features_labels.append((features, labels[offset]))
    if features_method == "simple-posseq":
        # simple features with posseq
        for offset, tag in enumerate(tags_segment):
            token_bc_start = max(offset - context_tokens, 0)
            features = simple_features(tag,
                                       context_start + offset,
                                       len(element["tags"]),
                                       tags_segment[token_bc_start:offset],
                                       tags_segment[offset+1:offset + context_tokens + 1]
                                      )
            posseq = " ".join([t[1] for t in element["tags"][start:end]])
            features.extend(['posseq=%s' % posseq])
            features_labels.append((features, labels[offset]))
    if features_method == "extended":
        # Extended features
        for offset, tag in enumerate(tags_segment):
            token_bc_start = max(offset - context_tokens, 0)
            features = extended_features(tag,
                                         context_start + offset,
                                         len(element["tags"]),
                                         tags_segment[token_bc_start:offset],
                                         tags_segment[offset+1:offset + context_tokens + 1]
                                        )
            features_labels.append((features, labels[offset]))
    return features_labels

def simple_features(tag, token_index, len_tags, beginning_context, ending_context):
    """Return simple features from token"""
    token, postag, _, _ = tag
    features = [
        'bias',
        'token.lower=%s' % token.lower(),
        'token.suffix[-3:]=%s' % token[-3:],
        'token.suffix[-2:]=%s' % token[-2:],
        'token.isupper=%s' % token.isupper(),
        'token.istitle=%s' % token.istitle(),
        'token.isdigit=%s' % token.isdigit(),
        'postag=%s' % postag,
        'postag[:2]=%s' % postag[:2]
        ]

    if token_index == 0:
        features.append('BOS')
    elif token_index == len_tags - 1:
        features.append('EOS')

    len_context = len(beginning_context)
    for i, (token, postag, _, _) in enumerate(beginning_context):
        context_index = len_context - i
        features.extend([
            '-%d:token.lower=%s' % (context_index, token.lower()),
            '-%d:token.istitle=%s' % (context_index, token.istitle()),
            '-%d:token.isupper=%s' % (context_index, token.isupper()),
            '-%d:postag=%s' % (context_index, postag),
            '-%d:postag[:2]=%s' % (context_index, postag[:2]),
            ])
    for context_index, (token, postag, _, _) in enumerate(ending_context, 1):
        features.extend([
            '+%d:token.lower=%s' % (context_index, token.lower()),
            '+%d:token.istitle=%s' % (context_index, token.istitle()),
            '+%d:token.isupper=%s' % (context_index, token.isupper()),
            '+%d:postag=%s' % (context_index, postag),
            '+%d:postag[:2]=%s' % (context_index, postag[:2]),
            ])
    return features

def extended_features(tag, token_index, len_tags, beginning_context, ending_context):
    """Return extended features from token"""
    token, _, _, _ = tag
    features = simple_features(tag,
                               token_index,
                               len_tags,
                               beginning_context,
                               ending_context
                              )
    features.extend([
        'token.suffix[:2]=%s' % token[:2],
        'token.suffix[:3]=%s' % token[:3]
    ])
    return features

def keyphrase_label_from(candidate_span, element, generic_label=True):
    """Receive candidate_span and element from dataset and return keyphrase label"""
    label = "NON-KEYPHRASE"
    if "keyphrases" in element and\
            "keyphrase-id" in candidate_span and \
            candidate_span["keyphrase-id"] in element["keyphrases"]:
        if not generic_label:
            label = element["keyphrases"][candidate_span["keyphrase-id"]]["keyphrase-label"]
        else:
            label = "KEYPHRASE"
    return label

def add_notation(tags_segment, label, left_context, right_context, tagging_notation="BILOU"):
    """Receive segment of tagged tokens and return list of labeled tokens"""
    labels = []
    if tagging_notation == "BIO":
        for offset in range(0, len(tags_segment)):
            if offset < left_context:
                token_label = "O"
            elif offset == left_context:
                token_label = "B-" + label
            elif offset < len(tags_segment) - right_context:
                token_label = "I-" + label if label else label
            else:
                token_label = "O"
            labels.append(token_label)
    if tagging_notation == "BILOU":
        for offset in range(0, len(tags_segment)):
            if offset < left_context:
                token_label = "O"
            elif offset == left_context and \
                    len(tags_segment) - left_context - right_context == 1:
                token_label = "U-" + label if label else label
            elif offset == left_context:
                token_label = "B-" + label
            elif offset < len(tags_segment) - right_context - 1:
                token_label = "I-" + label if label else label
            elif offset < len(tags_segment) - right_context:
                token_label = "L-" + label if label else label
            else:
                token_label = "O"
            labels.append(token_label)
    return labels

def largest_keyphrases(keyphrases):
    """Filter keyphrases choosing the largest"""
    keyphrases = sorted(keyphrases, key=lambda kp: (kp[1][1][0], -kp[1][1][1]))
    kps = keyphrases[:1]
    last = kps[0] if kps else None
    for i, keyphrase in enumerate(keyphrases[1:]):
        prev_start, prev_end = last[1][1]
        start, end = keyphrases[i+1][1][1]
        if not (prev_start <= start and prev_end >= end):
            last = keyphrase
            kps.append(last)
    return kps

def post_processing(keyphrases):
    """Post-process keyphrases list"""
    return largest_keyphrases(keyphrases)

def keyphrases2brat(keyphrases):
    """Receive keyphrases and return brat string"""
    brat_str = ""
    for keyphrase in keyphrases:
        keyphrase_id, (keyphrase_label_, (start, end)), keyphrase_str = keyphrase
        brat_str += "%s\t%s %s %s\t%s\n" % \
                    (keyphrase_id, keyphrase_label_, start, end, keyphrase_str)
    return brat_str.strip("\n")

def keyphrase_label(keyphrase):
    """Receive keyphrase and return label"""
    _, (keyphrase_label_, _), _ = keyphrase
    return keyphrase_label_

def keyphrase_span(keyphrase):
    """Receive keyphrase and return label"""
    _, (_, start_end), _ = keyphrase
    return start_end

def microaverage(spans, beta=1.0):
    """Receive list of spans [(list_extracted, list_annotated), ... ] 
    and return the micro-average (precision, recall, F1)"""
    beta2 = beta**2
    sum_tp = 0
    sum_tp_fp = 0
    sum_tp_fn = 0
    # For each result
    for selected, relevants in spans:
        # TP
        sum_tp += len(selected & relevants)
        # TP + FP
        sum_tp_fp += len(selected)
        # TP + FN
        sum_tp_fn += len(relevants)
    # Precision
    ma_precision = (sum_tp/sum_tp_fp) if sum_tp_fp > 0 else float('Inf')
    # Recall
    ma_recall = sum_tp/sum_tp_fn if sum_tp_fn > 0 else float('Inf')
    # F_1
    ma_f1 = (1 + beta2)*(ma_precision*ma_recall)/((beta2 * ma_precision) + ma_recall)
    return ma_precision, ma_recall, ma_f1