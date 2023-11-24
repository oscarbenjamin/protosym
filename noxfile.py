"""Nox sessions."""
import shutil
import sys
from pathlib import Path

import nox
from nox import Session
from nox import session


package = "protosym"
python_versions = ["3.9", "3.8"]
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = (
    "pre-commit",
    "mypy",
    "tests",
    "doctest",
    "docs-build",
)


@session(name="pre-commit", python="3.9")
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.install("-r", "requirements-lint.txt")
    session.run("pre-commit", *args)


@session(python="3.9")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    session.install(".")
    session.install("mypy", "pytest")
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install(".", "-r", "requirements-test.txt")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@session
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]

    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@session(python=python_versions)
def doctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.install(".", "-r", "requirements-test.txt")
    session.run("python", "-m", "xdoctest", "--quiet", package, *args)


@session(name="docs-build", python="3.9")
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["-W", "docs", "docs/_build"]
    session.install(".", "-r", "requirements-docs.txt")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python="3.9")
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build", "--watch=src"]
    session.install(".", "-r", "requirements-docs.txt")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
