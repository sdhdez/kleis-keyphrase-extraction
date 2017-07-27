from nose2.tools import such
from nose2.tools.params import params

from resources.items import get_files
from resources.items import get_files_by_ext

with such.A("module to load resources") as it:
    @it.has_setup
    def setup():
        pass

    @it.has_teardown
    def teardown():
        pass

    @it.should("get files")
    @params(".", "..")
    def test_get_files_basic(case, path):
        i = 0
        for f in get_files(path):
            i += 1
            case.assertEqual(type(f), str)
            break
        case.assertTrue(i > 0)

    @it.should("not get files")
    @params("why-this-work?", "I'm not a path.", "some_non_existing_path")
    def test_get_files_non_existing(case, path):
        i = 0
        for f in get_files(path):
            i += 1
            case.assertNotEqual(type(f), str)
        case.assertEqual(i, 0)

    @it.should("get files by extension")
    @params((".", "py"), 
            (".", "PY"), 
            ("..", "txt"))
    def test_get_files_by_ext(case, path, ext):
        i = 0
        for f in get_files_by_ext(path, ext = ext):
            i += 1
        case.assertTrue(i > 0)

    @it.should("not match an extension")
    @params((".", "pbc"), 
            (".", "SDF"), 
            ("..", "I'm not an extension."))
    def test_get_files_by_ext_not(case, path, ext):
        i = 0
        for f in get_files_by_ext(path, ext = ext):
            i += 1
            break
        case.assertEqual(i, 0)

it.createTests(globals())

if __name__ == "__main__":
    import nose2
    nose2.main()
