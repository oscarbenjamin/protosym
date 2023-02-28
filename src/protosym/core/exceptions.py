"""Module for all protosym exceptions."""


class ProtoSymError(Exception):
    """Superclass for all protosym exceptions."""

    pass


class NoEvaluationRuleError(ProtoSymError):
    """Raised when an :class:`Evaluator` has no rule for an expression."""

    pass
