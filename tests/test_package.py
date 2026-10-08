import spodnn
from importlib.metadata import version


def test_version_matches_metadata():
    assert spodnn.__version__ != ""
    assert spodnn.__version__ == version("spodnn")
