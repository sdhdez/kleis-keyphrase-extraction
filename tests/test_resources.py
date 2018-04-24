"""Testing module (nose2)"""
from pathlib import Path
import nose2
from nose2.tools import such
from nose2.tools.params import params

from resources import dataset as rd
from resources.semeval2017 import SemEval2017

with such.A("module to load resources") as it:
    @it.has_setup
    def setup():
        """Setup"""
        pass

    @it.has_teardown
    def teardown():
        """Teardown"""
        pass

    @it.should("get files in path")
    @params(".", "..")
    def test_get_files_basic(case, path):
        """Test to get files in path"""
        for filename in rd.get_files(path):
            case.assertTrue(issubclass(filename.__class__, Path))

    @it.should("not get files")
    @params("why-this-work?", "I'm not a path.", "some_non_existing_path")
    def test_get_files_non_existing(case, path):
        """Test wrong finle names"""
        for filename in rd.get_files(path):
            case.assertFalse(issubclass(filename.__class__, Path))

    @it.should("get files by extension")
    @params((".", "py", ".py"),
            (".", "PY", ".py"),
            ("..", "txt", ".txt"))
    def test_get_files_by_ext(case, path, ext, match_ext):
        """Test getting files by extension"""
        for filename in rd.get_files_by_ext(path, suffix=ext):
            case.assertTrue(issubclass(filename.__class__, Path))
            case.assertEqual(filename.suffix, match_ext)

    @it.should("not match an extension")
    @params((".", "pbc"),
            (".", "SDF"),
            ("..", "I'm not an extension."))
    def test_get_files_by_ext_not(case, path, ext):
        """Test behavior of trying to get wrong files"""
        for filename in rd.get_files_by_ext(path, suffix=ext):
            case.assertFalse(issubclass(filename.__class__, Path))

    @it.should("get content from files")
    def test_get_content():
        """Test getting file content"""
        for filename in rd.get_files_by_ext("..", suffix=".py"):
            content = rd.get_content(filename, suffixes=[".py"])
            it.assertTrue(content)

    @it.should("check if given directory paths do exist")
    @params(".", "..", "config/", "resources",
            "", "./tests/", "./corpus")
    def test_path_exists(case, path):
        """Test if given directory paths do exist"""
        case.assertTrue(rd.path_exists(path))
        case.assertTrue(rd.path_exists(Path(path)))

    @it.should("check if given directory paths do NOT exist")
    @params("df sdf sdf s dfs", "abcd dd", " s s d", True, 1, 0, "...")
    def test_path_exists_not(case, path):
        """Check if given directory paths do NOT exist"""
        case.assertFalse(rd.path_exists(path))
        if isinstance(path, str):
            case.assertFalse(rd.path_exists(Path(path)))

    @it.should("return valid paths in corpus config")
    @params(({"test0": ".", "test1": ".."}, 2),
            ({"test0": ".", "test1": "..", "test2": "../", "test3": None}, 3),
            ({"test0": 1, "test1": "QSDQSD", "test2": "", "test3": None}, 1))
    def test_get_corpus_paths(case, corpus_config, n_paths):
        """Test returning valid paths in corpus"""
        case.assertEqual(len(rd.get_corpus_paths(corpus_config)), n_paths)

    @it.should("load paths from SemEval2107Task10")
    def test_load_semeval2017task10():
        """Test loading paths from SemEval2107Task10"""
        semeval2017 = rd.load_config_corpus()
        it.assertEqual(len(semeval2017), 4)
        it.assertTrue("train-labeled" in semeval2017)
        it.assertFalse("train-unlabeled" in semeval2017)
        it.assertTrue("dev-labeled" in semeval2017)
        it.assertFalse("dev-unlabeled" in semeval2017)
        it.assertTrue("test-labeled" in semeval2017)
        it.assertTrue("test-unlabeled" in semeval2017)
        # Load train-labeled
        labeled_text = list(rd.get_files_by_ext(semeval2017['train-labeled'], suffix="txt"))
        it.assertEqual(len(labeled_text), 350)
        labeled_ann = list(rd.get_files_by_ext(semeval2017['train-labeled'], suffix="ann"))
        it.assertEqual(len(labeled_ann), 350)
        labeled_xml = list(rd.get_files_by_ext(semeval2017['train-labeled'], suffix="xml"))
        it.assertEqual(len(labeled_xml), 350)
        # Load dev-labeled
        labeled_text = list(rd.get_files_by_ext(semeval2017['dev-labeled'], suffix="txt"))
        it.assertEqual(len(labeled_text), 50)
        labeled_ann = list(rd.get_files_by_ext(semeval2017['dev-labeled'], suffix="ann"))
        it.assertEqual(len(labeled_ann), 50)
        labeled_xml = list(rd.get_files_by_ext(semeval2017['dev-labeled'], suffix="xml"))
        it.assertEqual(len(labeled_xml), 50)
        # Load test-labeled
        labeled_text = list(rd.get_files_by_ext(semeval2017['test-labeled'], suffix="txt"))
        it.assertEqual(len(labeled_text), 100)
        labeled_ann = list(rd.get_files_by_ext(semeval2017['test-labeled'], suffix="ann"))
        it.assertEqual(len(labeled_ann), 100)
        # Load test-unlabeled
        unlabeled_text = list(rd.get_files_by_ext(semeval2017['test-unlabeled'], suffix="txt"))
        it.assertEqual(len(unlabeled_text), 100)
        unlabeled_xml = list(rd.get_files_by_ext(semeval2017['test-unlabeled'], suffix="xml"))
        it.assertEqual(len(unlabeled_xml), 100)

    @it.should("fail to load a wrong corpus config")
    @params("some-name", {"dataset": "."}, 5, [1, 3], "")
    def test_wrong_semeval2017task10(case, corpus_name):
        """Test try to load a false corpus config"""
        case.assertEqual(rd.load_config_corpus(name=corpus_name), None)

    @it.should("load raw documents from dataset")
    def test_load_dataset():
        """Test loading default dataset"""
        semeval2017 = rd.load_config_corpus()
        labeled = dict(rd.load_dataset_raw(semeval2017['train-labeled'],
                                           [".txt", ".ann", ".xml"],
                                           suffix="txt"))
        it.assertEqual(len(labeled), 350)

    @it.should("fail to load raw documents from dataset")
    def test_not_load_dataset():
        """Test not loading default dataset"""
        semeval2017 = rd.load_config_corpus()
        labeled = dict(rd.load_dataset_raw(semeval2017['train-labeled'],
                                           [".some", ".ext_", ".notext"],
                                           suffix="txt"))
        it.assertEqual(len(labeled), 350)

    @it.should("load default corpus (SemEval2107Task10)")
    def test_load_default_curpus():
        """Test loading default corpus"""
        default_corpus = rd.load_corpus()
        it.assertTrue(isinstance(default_corpus, SemEval2017))
        it.assertTrue(isinstance(default_corpus.config, dict))
        it.assertEqual(len(default_corpus.train), 350)
        it.assertEqual(len(default_corpus.train.popitem()[1]['raw']), 3)
        it.assertEqual(len(default_corpus.dev), 50)
        it.assertEqual(len(default_corpus.train.popitem()[1]['raw']), 3)
        it.assertEqual(len(default_corpus.test), 100)
        it.assertEqual(len(default_corpus.train.popitem()[1]['raw']), 3)

    @it.should("tokenize English sentences")
    @params("Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.")
    def test_tokenize_english(case, text):
        """Test tokenizing text in English"""
        tokens, tokens_span = rd.tokenize_en(text)
        for i, (start, end) in enumerate(tokens_span):
            case.assertEqual(text[start:end], tokens[i])

    @it.should("tag tokens")
    @params("Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.")
    def test_tag_text_en(case, text):
        """Test tagging English text"""
        tokens, tokens_span = rd.tokenize_en(text)
        tags = rd.tag_text_en(tokens, tokens_span)
        # Expected tokens
        case.assertEqual(len(tags), 18)
        # Expected fields
        case.assertEqual(len(tags[0]), 4)
        # Token
        case.assertIsInstance(tags[0][0], str)
        # PoSTag
        case.assertIsInstance(tags[0][1], str)
        # Span
        case.assertIsInstance(tags[0][2], tuple)
        # (start, end)
        case.assertIsInstance(len(tags[0][2]), 2)
        # List of terms
        case.assertIsInstance(tags[0][3], list)

it.createTests(globals())

if __name__ == "__main__":
    nose2.main()
