"""
HFA AST base classes
"""

import abc


class Argument(metaclass=abc.ABCMeta):
    """
    Argument interface
    """

    @abc.abstractmethod
    def gen(self) -> str:
        """
        Generate the HFA code for this Argument
        """
        raise NotImplementedError  # pragma: no cover


class Expression(metaclass=abc.ABCMeta):
    """
    Expression interface
    """

    @abc.abstractmethod
    def gen(self) -> str:
        """
        Generate the HFA code for this Expression
        """
        raise NotImplementedError  # pragma: no cover


class Operator(metaclass=abc.ABCMeta):
    """
    Operator interface
    """

    @abc.abstractmethod
    def gen(self) -> str:
        """
        Generate the HFA code for this Operator
        """
        raise NotImplementedError  # pragma: no cover


class Payload(metaclass=abc.ABCMeta):
    """
    Payload interface
    """

    @abc.abstractmethod
    def gen(self, parens: bool) -> str:
        """
        Generate the HFA code for this Payload
        """
        raise NotImplementedError  # pragma: no cover


class Statement(metaclass=abc.ABCMeta):
    """
    Statement interface
    """

    @abc.abstractmethod
    def gen(self, depth: int) -> str:
        """
        Generate the HFA code for this Statement
        """
        raise NotImplementedError  # pragma: no cover
