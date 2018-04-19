"""main

Execute methods.

"""
import resources.dataset as rd

def main():
    """Method to run package."""
    # Load default dataset
    default_dataset = rd.load_corpus()
    print("Name:", default_dataset.name)
    print("Config:", default_dataset.config)
    print("len(Train):", len(default_dataset.train))
    # print("Train:", default_dataset.train.popitem())
    print("len(Dev):", len(default_dataset.dev))
    # print("Dev:", default_dataset.dev.popitem())
    print("len(Test):", len(default_dataset.test))
    # print("Test:", default_dataset.test.popitem())

if __name__ == "__main__":
    main()
