#!/bin/bash
#
# Check if any pinned versions can be updated.

pip-upgrade requirements-docs.txt
pip-upgrade requirements-lint.txt
pip-upgrade requirements-test.txt
pip-upgrade requirements-all.txt
