from nose2.tools import such
from nose2.tools.params import params

from resources.items import get_files

with such.A("module to load resources") as it:
    @it.has_setup
    def setup():
        pass

    @it.has_teardown
    def teardown():
        pass

    @it.should("get_files from corpus")
    def test_get_files_basic():
        for f in get_files(".."):
            it.assert_(type(f), str)

it.createTests(globals())

if __name__ == "__main__":
    import nose2
    nose2.main()
