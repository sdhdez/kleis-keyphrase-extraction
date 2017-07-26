from nose2.tools import such
from nose2.tools.params import params

from resources.items import get_files

with such.A("Load 'resources'") as it:

    @it.has_setup
    def setup():
        pass

    @it.has_teardown
    def teardown():
        pass

    @it.should("get_files from corpus")
    def test_get_files_basic(self):
        for f in get_files(".."):
            it.assertEqual(type(f), str)

if __name__ == "__main__":
    import nose2
    nose2.main()
