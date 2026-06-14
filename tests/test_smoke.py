import vllmstat


def test_version_present():
    assert isinstance(vllmstat.__version__, str)
    assert vllmstat.__version__.count(".") >= 2
