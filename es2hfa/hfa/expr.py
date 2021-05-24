"""
HFA AST and code generation for HFA expressions
"""

import abc


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


@Expression.register
class EVar:
    """
    An HFA variable
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def gen(self) -> str:
        """
        Generate the HFA code for an EVar
        """
        return self.name
