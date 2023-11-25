#!/bin/bash

set -o errexit

# # Clone rust_protosym in parallel directory:
#
# git clone https://github.com/oscarbenjamin/rust_protosym.git ../rust_protosym

if [ "$1" = "--rs" ]; then
    # Rebuild and reinstall rust_protosym:
    cd ../rust_protosym
      maturin build -r
      pip install target/wheels/*.whl  --force-reinstall
    cd -
fi

hatch run pre-commit:run
hatch run types:mypy-check
hatch run docs:build
hatch run test:doctest
hatch run test:coverage
