"""Exception types raised in simplecas."""
from protosym.core.exceptions import ProtoSymError


class SimpleCASError(ProtoSymError):
    """Base class for all exceptions in simplecas."""

    pass


class ExpressifyError(SimpleCASError, TypeError):
    """Raised when an object cannot be expressified."""

    pass


class LLVMNotImplementedError(ProtoSymError):
    """Raised when an operation is not supported for LLVM."""

    pass
