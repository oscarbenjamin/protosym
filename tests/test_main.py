"""Test cases for the __main__ module."""
from protosym import __main__


def test_main_succeeds() -> None:
    """It exits with a status code of zero."""
    assert __main__.main() == 0
