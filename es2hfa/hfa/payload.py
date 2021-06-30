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

    def gen(self, parens: bool) -> str:
        """
        Generate the HFA output for an SBlock
        """
        payload = ", ".join([p.gen(True) for p in self.payloads])
        if parens:
            return "(" + payload + ")"
        return payload


@Payload.register
class PVar:
    """
    A single variable payload
    """

    def __init__(self, var: str) -> None:
        self.var = var

    def gen(self, parens: bool) -> str:
        """
        Generate the HFA output for an SBlock

        Note: the parens argument has no impact on PVar because it is already
        an atomic element
        """
        return self.var
