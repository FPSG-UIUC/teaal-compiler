"""
MIT License

Copyright (c) 2021 University of Illinois

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Useful functions for generating HiFiber code
"""

from copy import deepcopy

from typing import Any

from teaal.hifiber import *
from teaal.ir.tensor import Tensor


class TransUtils:
    """
    Different utilities for generating HiFiber programs
    """

    def __init__(self) -> None:
        self.count = -1

    @staticmethod
    def build_expr(obj: Any) -> Expression:
        """
        Build an HiFiber expression for the given Python object
        """
        if isinstance(obj, int):
            return EInt(obj)

        elif isinstance(obj, float):
            return EFloat(obj)

        elif isinstance(obj, str):
            return EString(obj)

        elif isinstance(obj, list):
            list_ = [TransUtils.build_expr(elem) for elem in obj]
            return EList(list_)

        elif isinstance(obj, dict):
            dict_ = {TransUtils.build_expr(key): TransUtils.build_expr(val)
                     for key, val in obj.items()}
            return EDict(dict_)

        else:
            raise ValueError("Unable to translate " +
                             str(obj) + " with type " + str(type(obj)))

    @staticmethod
    def build_rank_ids(tensor: Tensor) -> Argument:
        """
        Build the rank_ids argument
        """
        ranks = [EString(rank) for rank in tensor.get_ranks()]
        return AParam("rank_ids", EList(ranks))

    @staticmethod
    def build_set_rank_ids(tensor: Tensor, name: str) -> Statement:
        """
        Build the setRankIds() function
        """
        arg = TransUtils.build_rank_ids(tensor)
        set_call = EMethod(EVar(name), "setRankIds", [arg])
        return SExpr(set_call)

    @staticmethod
    def build_shape(tensor: Tensor) -> Argument:
        """
        Build the shape argument
        """
        ranks = [EVar(rank) for rank in tensor.get_ranks()]
        return AParam("shape", EList(ranks))

    @staticmethod
    def build_swizzle(
            tensor: Tensor,
            old_name: str,
            new_name: str) -> Statement:
        """
        Build the swizzleRanks() function
        """
        arg = TransUtils.build_rank_ids(tensor)
        swizzle_call = EMethod(EVar(old_name), "swizzleRanks", [arg])
        new_name_assn = AVar(new_name)
        return SAssign(new_name_assn, swizzle_call)

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

    @staticmethod
    def sub_hifiber(hifiber: Base, old: Base, new: Base) -> Base:
        """
        Substitute all instances of old with new in hifiber

        Note: the type checking is not strict enough to ensure correctness
        """
        if hifiber == old:
            return deepcopy(new)

        copied = deepcopy(hifiber)
        attrs = vars(copied)
        for key, val in attrs.items():
            if isinstance(val, Base):
                copied.__dict__[key] = TransUtils.sub_hifiber(val, old, new)

        return copied
