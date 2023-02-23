#!/bin/bash

poetry run pre-commit run --all-files
poetry run mypy src tests
poetry run pytest
poetry run python -m xdoctest --quiet protosym
