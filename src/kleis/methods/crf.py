"""methods/crf

    CRF module

"""
import os
import pycrfsuite

from kleis.config.config import MODELS_PATH
from kleis.resources import dataset as kl

def crf_preprocess_candidates(candidates):
    """Receive annotated candidates and return features and labels list"""
    features = []
    labels = []
    for candidate in candidates:
        candidate_features = []
        candidate_labels = []
        for token_features, label in candidate:
            candidate_features.append(token_features)
            candidate_labels.append(label)
        features.append(candidate_features)
        labels.append(candidate_labels)
    return features, labels

def pycrfsuite_train(annotated_candidates, name="candidates-model.pycrfsuite"):
    """Receive annotated candidates and train model"""
    if not kl.path_exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    model = MODELS_PATH + name
    if not kl.path_exists(model):
        features, labels = [], []
        for candidates in annotated_candidates.values():
            candidate_features, candidate_labels = crf_preprocess_candidates(candidates)
            features.extend(candidate_features)
            labels.extend(candidate_labels)
        # pycrfsuite
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(features, labels):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            # 'max_iterations': 50,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.params()
        trainer.train(model)
    tagger = pycrfsuite.Tagger()
    tagger.open(model)
    return tagger

def pycrfsuite_label(tagger, pos_sequences, text, context_tokens=1,
                     features_method="simple", tagging_notation="BILOU", generic_label=True):
    """Receive tagger, pos sequences and text and return labeled text"""
    tokens, tokens_span = kl.tokenize_en(text)
    tags = kl.tag_text_en(tokens, tokens_span)
    dataset_element_fake = {"tags": tags}
    candidates_spans = kl.filter_pos_sequences(
        dataset_element_fake,
        pos_sequences,
        annotated=False
    )
    candidates = kl.candidates_spans_features_labels_from(
        candidates_spans, dataset_element_fake,
        context_tokens=context_tokens,
        features_method=features_method,
        tagging_notation=tagging_notation,
        generic_label=generic_label
    )
    candidates_features, _ = crf_preprocess_candidates(candidates)
    keyphrases = []
    for i, candidate_feaures in enumerate(candidates_features):
        labeled_candidate = tagger.tag(candidate_feaures)
        if is_keyphrase((labeled_candidate, candidates_spans[i]),
                        tags, pos_sequences, tagging_notation=tagging_notation):
            keyphrase_label_span = labeled_keyphrase_span(
                (labeled_candidate, candidates_spans[i]),
                tags,
                tagging_notation=tagging_notation
            )
            keyphrase_label, (keyphrase_span_start, keyphrase_span_end) = keyphrase_label_span
            keyphrases.append(
                ("T%d" % (i + 1),
                 (keyphrase_label, (keyphrase_span_start, keyphrase_span_end)),
                 text[keyphrase_span_start:keyphrase_span_end])
            )
    return keyphrases

def is_keyphrase(labeled_candidate, tags, pos_sequences, tagging_notation="BILOU"):
    """Receive labeled candidate and return true or false"""
    labels, candidate_spans = labeled_candidate
    start, end = candidate_spans["span"]
    expected_tokens = end - start
    is_valid = False
    if tagging_notation == "BIO" or tagging_notation == "BILOU":
        postags = list(map(lambda t: t[1], tags[start:end]))
        labels_valid = list(map(lambda l: l[2:],
                                filter(lambda l: l != "O" \
                                       and l[-len("NON-KEYPHRASE"):] != "NON-KEYPHRASE",
                                       labels)))
        if len(labels_valid) == expected_tokens \
                and postags == pos_sequences[candidate_spans["pos-seq-id"]]["tags"] \
                and len(set(labels_valid)) == 1:
            is_valid = True
    return is_valid

def labeled_keyphrase_span(keyphrase, tags, tagging_notation="BILOU"):
    """Receive labeled keyphrase and return span"""
    labeled_candidate, candidate_spans = keyphrase
    start, end = candidate_spans["span"]
    label = "KEYPHRASE"
    if tagging_notation == "BIO" or tagging_notation == "BILOU":
        label = list(set(list(filter(lambda lc: lc != "O", labeled_candidate))))[0][2:]
    _, _, token_span_start, _ = tags[start]
    _, _, token_span_end, _ = tags[end - 1]
    span = (token_span_start[0], token_span_end[1])
    return label, span
