"""
HFA AST and code generation for HFA statements
"""

import abc
from typing import List

from es2hfa.hfa.base import Statement


@Statement.register
class SBlock:
    """
    A block of statements
    """

    def __init__(self, stmts: List[str]) -> None:
        self.stmts = stmts

    def gen(self) -> str:
        """
        Generate the HFA output for an SBlock
        """
        return "\n".join(self.stmts)
