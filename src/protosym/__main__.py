"""Command-line interface."""
import sys


def main(*args: str) -> int:
    """ProtoSym."""
    print("Welcome to ProtoSym!")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))  # pragma: no cover
