#!/bin/bash

poetry run mypy src
poetry run pytest
