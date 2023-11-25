#!/bin/bash

pre-commit run --all-files
mypy src tests
python -m xdoctest --quiet protosym
pytest --cov=protosym
coverage html
