from nose2.tools import such
from nose2.tools.params import params

from resources import dataset as resources_dataset

with such.A("module to load resources") as it:
    @it.has_setup
    def setup():
        pass

    @it.has_teardown
    def teardown():
        pass

    @it.should("get files in path")
    @params(".", "..")
    def test_get_files_basic(case, path):
        i = 0
        for fname in resources_dataset.get_files(path):
            i += 1
            case.assertEqual(type(fname), str)
        else:
            case.assertGreater(i, 0)

    @it.should("not get files")
    @params("why-this-work?", "I'm not a path.", "some_non_existing_path")
    def test_get_files_non_existing(case, path):
        i = 0
        for fname in resources_dataset.get_files(path):
            i += 1
            case.assertNotEqual(type(fname), str)
        else:
            case.assertEqual(i, 0)

    @it.should("get files by extension")
    @params((".", "py", ".py", 3), 
            (".", "PY", ".py", 3), 
            ("..", "txt", ".txt", 4))
    def test_get_files_by_ext(case, path, ext, match_ext, len_ext):
        i = 0
        for fname in resources_dataset.get_files_by_ext(path, ext = ext):
            i += 1
            case.assertEqual(type(fname), str)
            case.assertEqual(fname[-len_ext:].lower(), match_ext)
        else:
            case.assertGreater(i, 0)

    @it.should("not match an extension")
    @params((".", "pbc"), 
            (".", "SDF"), 
            ("..", "I'm not an extension."))
    def test_get_files_by_ext_not(case, path, ext):
        i = 0
        for fname in resources_dataset.get_files_by_ext(path, ext = ext):
            i += 1
            case.assertNotEqual(type(fname), str)
        else:
            case.assertEqual(i, 0)

    @it.should("check if given directory paths do exist")
    @params(".", "..", "config/", "resources", "./tests/", "./corpus")
    def test_is_path_here(case, path):
        case.assertTrue(resources_dataset.is_path_here(path))

    @it.should("check if given directory paths do NOT exist")
    @params("df sdf sdf s dfs", "abcd dd", " s s d", True, 1, 0, "...", "")
    def test_is_path_here_not(case, path):
        case.assertFalse(resources_dataset.is_path_here(path))

    @it.should("return valid paths in dataset")
    @params(({"test0": ".", "test1": ".."}, 2),
                ({"test0": ".", "test1": "..", "test2": "../", "test3": None}, 3),
                ({"test0": 1, "test1": "QSDQSD", "test2": "", "test3": None}, 0))
    def test_get_dataset_paths(case, dataset, n_paths):
        case.assertEqual(len(resources_dataset.get_dataset_paths(dataset)), n_paths)

    @it.should("load paths from SemEval2107Task10")
    def test_load_dataset_semeval2017task10():
        semeval2017 = resources_dataset.load_dataset_semeval2017task10()
        it.assertEqual(len(semeval2017), 4)
        it.assertTrue("train-labeled" in semeval2017)
        it.assertFalse("train-unlabeled" in semeval2017)
        it.assertTrue("dev-labeled" in semeval2017)
        it.assertFalse("dev-unlabeled" in semeval2017)
        it.assertTrue("test-labeled" in semeval2017)
        it.assertTrue("test-unlabeled" in semeval2017)

    @it.should("try to load a false dataset")
    @params({}, None, {"dataset": None}, {"dataset": "."}, 5, [], [1, 3])
    def test_load_dataset_semeval2017task10(case, dataset):
        case.assertEqual(resources_dataset.load_dataset_semeval2017task10(corpus=dataset), None)

it.createTests(globals())

if __name__ == "__main__":
    import nose2
    nose2.main()
