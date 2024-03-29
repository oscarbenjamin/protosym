#!/bin/bash

set -o errexit

pre-commit run --all-files
mypy --python-version=3.12 src tests
sphinx-build -b html docs docs/_build
python -m xdoctest --quiet protosym
pytest --cov=protosym
coverage html
