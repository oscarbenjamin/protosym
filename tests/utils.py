# Utilities for the test suite
from typing import Any, Callable

__all__ = [
    "requires",
    "requires_llvmlite",
    "requires_numpy",
    "requires_sympy",
]

from functools import wraps

import pytest


def requires(module_name: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Return a decorator to skip tests if module is not available."""

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            try:
                __import__(module_name)
            except ImportError:
                pytest.skip("requires %s" % module_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


requires_llvmlite = requires("llvmlite")
requires_numpy = requires("numpy")
requires_sympy = requires("sympy")
