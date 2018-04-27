"""main

Execute methods.

"""
import resources.dataset as rd

def main():
    """Method to run package."""
    # Load default dataset
    default_corpus = rd.load_corpus()
    print("Name:", default_corpus.name)
    print("Config:", default_corpus.config)
    print("len(Train):", len(default_corpus.train))
    # print("Train:", default_corpus.train.popitem())
    print("len(Dev):", len(default_corpus.dev))
    # print("Dev:", default_corpus.dev.popitem())
    print("len(Test):", len(default_corpus.test))
    # print("Test:", default_corpus.test.popitem())

    key = list(default_corpus.test.keys())[1]
    text = default_corpus.test[key]["raw"]["txt"]

    print("Document example...\n")
    print("Name of document:", key)
    print("Content to label:\n\n", text)

    # Train or load model
    default_corpus.training(filter_min_count=3)

    # Labeling
    keyphrases = default_corpus.label_text(text)
    print("Example of labeled keyphrases:\n\n", keyphrases)
    print("\nKeyphrases in %s.ann:\n" % key)
    print(rd.keyphrases2brat(keyphrases))

if __name__ == "__main__":
    main()
