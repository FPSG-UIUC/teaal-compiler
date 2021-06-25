"""
Useful functions for generating HFA code
"""

from typing import cast

from es2hfa.hfa.arg import AParam
from es2hfa.hfa.base import Argument, Expression, Statement
from es2hfa.hfa.expr import EList, EMethod, EString
from es2hfa.hfa.stmt import SAssign, SExpr
from es2hfa.ir.tensor import Tensor


class Utils:
    """
    Different utilities for generating HFA programs
    """

    def __init__(self) -> None:
        self.count = -1

    @staticmethod
    def build_rank_ids(tensor: Tensor) -> Argument:
        """
        Build the rank_ids argument
        """
        ranks = [cast(Expression, EString(ind)) for ind in tensor.get_inds()]
        arg = AParam("rank_ids", cast(Expression, EList(ranks)))
        return cast(Argument, arg)

    @staticmethod
    def build_set_rank_ids(tensor: Tensor) -> Statement:
        """
        Build the setRankIds() function
        """
        arg = Utils.build_rank_ids(tensor)
        set_call = EMethod(tensor.tensor_name(), "setRankIds", [arg])
        return cast(Statement, SExpr(cast(Expression, set_call)))

    @staticmethod
    def build_swizzle(tensor: Tensor, old_name: str) -> Statement:
        """
        Build the swizzleRanks() function
        """
        arg = Utils.build_rank_ids(tensor)
        swizzle_call = cast(
            Expression,
            EMethod(
                old_name,
                "swizzleRanks",
                [arg]))
        return cast(Statement, SAssign(tensor.tensor_name(), swizzle_call))

    def curr_tmp(self) -> str:
        """
        Get the last temporary returned
        """
        if self.count == -1:
            raise ValueError("No previous temporary")

        return "tmp" + str(self.count)

    def next_tmp(self) -> str:
        """
        Get a new unique temporary
        """
        self.count += 1
        return "tmp" + str(self.count)
