"""
HFA AST base classes
"""

import abc


class Argument(metaclass=abc.ABCMeta):
    """
    Argument interface
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Attributes a Argument must have
        """
        return (hasattr(subclass, "gen"))

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
    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Attributes a Expression must have
        """
        return (hasattr(subclass, "gen"))

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
    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Attributes a Operator must have
        """
        return (hasattr(subclass, "gen"))

    @abc.abstractmethod
    def gen(self) -> str:
        """
        Generate the HFA code for this Operator
        """
        raise NotImplementedError  # pragma: no cover


class Statement(metaclass=abc.ABCMeta):
    """
    Statement interface
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Attributes a Statement must have
        """
        return (hasattr(subclass, "gen"))

    @abc.abstractmethod
    def gen(self) -> str:
        """
        Generate the HFA code for this Statement
        """
        raise NotImplementedError  # pragma: no cover
