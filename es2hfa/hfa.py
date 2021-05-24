import abc
from typing import List

class Statement(metaclass=abc.ABCMeta):
    """
    Statement interface

    Note that right now
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "generate"))

@Statement.register
class SBlock:
    def __init__(self, stmts: List[str]) -> None:
        self.stmts = stmts

    def generate(self) -> str:
        return "\n".join(self.stmts)
