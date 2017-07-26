import os

def get_files(path_to_files, match = None, exclude = None):
    for dirpath, dirnames, filenames in os.walk(path_to_files):
        for fname in filenames:
            yield dirpath + " " + fname
