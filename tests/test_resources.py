"""Testing module (nose2)"""
import nose2
from nose2.tools import such
from nose2.tools.params import params

from resources import dataset as rd

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
        i = 0
        for fname in rd.get_files(path):
            i += 1
            case.assertEqual(type(fname), str)

    @it.should("not get files")
    @params("why-this-work?", "I'm not a path.", "some_non_existing_path")
    def test_get_files_non_existing(case, path):
        """Test wrong finle names"""
        i = 0
        for fname in rd.get_files(path):
            i += 1
            case.assertNotEqual(type(fname), str)

    @it.should("get files by extension")
    @params((".", "py", ".py", 3),
            (".", "PY", ".py", 3),
            ("..", "txt", ".txt", 4))
    def test_get_files_by_ext(case, path, ext, match_ext, len_ext):
        """Test getting files by extension"""
        i = 0
        for fname in rd.get_files_by_ext(path, ext=ext):
            i += 1
            case.assertEqual(type(fname), str)
            case.assertEqual(fname[-len_ext:].lower(), match_ext)

    @it.should("not match an extension")
    @params((".", "pbc"),
            (".", "SDF"),
            ("..", "I'm not an extension."))
    def test_get_files_by_ext_not(case, path, ext):
        """Test behavior of trying to get wrong files"""
        i = 0
        for fname in rd.get_files_by_ext(path, ext=ext):
            i += 1
            case.assertNotEqual(type(fname), str)

    @it.should("check if given directory paths do exist")
    @params(".", "..", "config/", "resources", "./tests/", "./corpus")
    def test_path_exists(case, path):
        """Test if given directory paths do exist"""
        case.assertTrue(rd.path_exists(path))

    @it.should("check if given directory paths do NOT exist")
    @params("df sdf sdf s dfs", "abcd dd", " s s d", True, 1, 0, "...", "")
    def test_path_exists_not(case, path):
        """Check if given directory paths do NOT exist"""
        case.assertFalse(rd.path_exists(path))

    @it.should("return valid paths in dataset")
    @params(({"test0": ".", "test1": ".."}, 2),
            ({"test0": ".", "test1": "..", "test2": "../", "test3": None}, 3),
            ({"test0": 1, "test1": "QSDQSD", "test2": "", "test3": None}, 0))
    def test_get_dataset_paths(case, dataset, n_paths):
        """Test returning valid paths in dataset"""
        case.assertEqual(len(rd.get_dataset_paths(dataset)), n_paths)

    @it.should("load paths from SemEval2107Task10")
    def test_load_semeval2017task10():
        """Test loading paths from SemEval2107Task10"""
        semeval2017 = rd.load_dataset_semeval2017task10()
        it.assertEqual(len(semeval2017), 4)
        it.assertTrue("train-labeled" in semeval2017)
        it.assertFalse("train-unlabeled" in semeval2017)
        it.assertTrue("dev-labeled" in semeval2017)
        it.assertFalse("dev-unlabeled" in semeval2017)
        it.assertTrue("test-labeled" in semeval2017)
        it.assertTrue("test-unlabeled" in semeval2017)

    @it.should("fail to load a wrong dataset config")
    @params({}, {"dataset": None}, {"dataset": "."}, 5, [], [1, 3])
    def test_wrong_semeval2017task10(case, dataset):
        """Test try to load a false dataset"""
        case.assertEqual(rd.load_dataset_semeval2017task10(corpus=dataset), None)

it.createTests(globals())

if __name__ == "__main__":
    nose2.main()
