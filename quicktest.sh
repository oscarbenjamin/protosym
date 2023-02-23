#!/bin/bash

poetry run pre-commit run --all-files
poetry run mypy src tests
poetry run python -m xdoctest --quiet protosym
poetry run pytest --cov=protosym
poetry run coverage html
