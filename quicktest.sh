#!/bin/bash

poetry run mypy src tests
poetry run pytest
