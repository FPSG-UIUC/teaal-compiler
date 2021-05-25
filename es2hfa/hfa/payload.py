"""
HFA AST and code generation for HFA payloads (the output of iterating over fibers)
"""

from typing import List

from es2hfa.hfa.base import Payload


@Payload.register
class PTuple:
    """
    A tuple of payloads
    """

    def __init__(self, payloads: List[Payload]) -> None:
        self.payloads = payloads

    def gen(self) -> str:
        """
        Generate the HFA output for an SBlock
        """
        return "(" + ", ".join([p.gen() for p in self.payloads]) + ")"


@Payload.register
class PVar:
    """
    A single variable payload
    """

    def __init__(self, var: str) -> None:
        self.var = var

    def gen(self) -> str:
        """
        Generate the HFA output for an SBlock
        """
        return self.var
