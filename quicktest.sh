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

pre-commit run --all-files
mypy src tests
python -m xdoctest --quiet protosym
pytest --cov=protosym
coverage html
