#!/bin/bash
#
# Check if any pinned versions can be updated.

# Checks pyproject.toml. The changes can be applied automatically with:
#
#   $ poetry up
#
# That will update pyproject.toml and poetry.lock for the dependencies
# explicitly listed in pyproject.toml.
#
# Requires poetry-plugin-up
poetry up --dry-run | diff -Nurp pyproject.toml -

# XXX: The above ouputs an unecessary blank line change in the diff that should
# be ignored. The commands below will enter an interactive prompt. Enter "x" to
# exit or "all" to apply changes in place.

poetry run pip-upgrade .github/workflows/constraints.txt
poetry run pip-upgrade docs/requirements.txt
