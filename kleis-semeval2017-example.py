"""main

Execute methods.

"""
import kleis.resources.dataset as kl

def main():
    """Method to run package."""
    # Load default dataset
    default_corpus = kl.load_corpus()
    print("Name:", default_corpus.name)
    print("Config:", default_corpus.config)
    print("len(Train):", len(default_corpus.train) if default_corpus.train else None)
    # print("Train:", default_corpus.train.popitem())
    print("len(Dev):", len(default_corpus.dev) if default_corpus.dev else None)
    # print("Dev:", default_corpus.dev.popitem())
    print("len(Test):", len(default_corpus.test) if default_corpus.test else None)
    # print("Test:", default_corpus.test.popitem())

    text = """Information extraction is the process of extracting structured \
data from unstructured text, which is relevant for several end-to-end tasks, \
including question answering. \
This paper addresses the tasks of named entity recognition (NER), \
a subtask of information extraction, using conditional random fields (CRF). \
Our method is evaluated on the ConLL-2003 NER corpus.
"""

    print("Document example...\n")
    print("Content to label:\n\n", text)

    # Train or load model
    default_corpus.training(filter_min_count=3)

    # Labeling
    keyphrases = default_corpus.label_text(text)
    print("Example of labeled keyphrases:\n\n", keyphrases)
    print(kl.keyphrases2brat(keyphrases))

if __name__ == "__main__":
    main()
