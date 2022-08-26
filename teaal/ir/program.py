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

Representation of einsum metadata and the specification
"""

from collections import Counter

from lark.tree import Tree
from typing import Dict, List, Optional, Set

from teaal.ir.coord_math import CoordMath
from teaal.ir.loop_order import LoopOrder
from teaal.ir.partitioning import Partitioning
from teaal.ir.spacetime import SpaceTime
from teaal.ir.tensor import Tensor
from teaal.parse.einsum import Einsum
from teaal.parse.mapping import Mapping
from teaal.parse.utils import ParseUtils


class Program:
    """
    Store tensor metadata and configure the metadata for specific einsums
    """

    def __init__(self, einsum: Einsum, mapping: Mapping) -> None:
        """
        Construct the metadata for tensors and prepare for an einsum
        """
        self.einsum = einsum
        self.mapping = mapping

        # Get all tensors
        self.decl_tensors = {}
        declaration = self.einsum.get_declaration()
        for ten_name in declaration:
            tensor = Tensor(ten_name, declaration[ten_name])
            self.decl_tensors[tensor.root_name()] = tensor

        # Replace the tensors whose rank order is specified
        self.tensors = {}
        rank_orders = self.mapping.get_rank_orders()

        for ord_name, tensor in self.decl_tensors.items():
            if ord_name in rank_orders.keys():
                tensor = Tensor(ord_name, rank_orders[ord_name])
            else:
                tensor = Tensor(ord_name, tensor.get_ranks())

            self.tensors[tensor.root_name()] = tensor

        self.einsum_ind: Optional[int] = None
        self.equation: Optional[Tree] = None
        self.es_tensors: List[Tensor] = []
        self.coord_math: Optional[CoordMath] = None
        self.loop_order: Optional[LoopOrder] = None
        self.partitioning: Optional[Partitioning] = None
        self.spacetime: Optional[SpaceTime] = None

    def add_einsum(self, i: int) -> None:
        """
        Configure the program for the i'th Einsum
        """
        self.einsum_ind = i
        self.equation = self.einsum.get_expressions()[i]
        self.coord_math = CoordMath()

        # Build the list of tensors, starting with the output tensor
        self.es_tensors = []
        output_tree = next(self.equation.find_data("output"))
        output = self.__get_tensor(output_tree)
        output.set_is_output(True)
        self.es_tensors.append(output)
        self.__add_tranks(output, output_tree)

        # Add the rest of the tensors
        for tensor_tree in self.equation.find_data("tensor"):
            self.es_tensors.append(self.__get_tensor(tensor_tree))
            self.__add_tranks(self.es_tensors[-1], tensor_tree)

        # Create the loop_order object
        self.loop_order = LoopOrder(self.equation, output)

        # Store the partitioning information
        partitioning = self.mapping.get_partitioning()
        ranks = self.__all_ranks()
        if output.root_name() in partitioning.keys():
            self.partitioning = Partitioning(
                partitioning[output.root_name()], ranks, self.coord_math.get_eqn_exprs())
        else:
            self.partitioning = Partitioning(
                {}, ranks, self.coord_math.get_eqn_exprs())

        # Store the loop order
        loop_orders = self.mapping.get_loop_orders()
        opt_loop_order: Optional[List[str]] = None
        if output.root_name() in loop_orders.keys():
            opt_loop_order = loop_orders[output.root_name()]

        self.loop_order.add(opt_loop_order, self.coord_math, self.partitioning)

        # Prune the coord math with this loop order
        self.coord_math.prune(self.loop_order.get_ranks(), self.partitioning)

        # Get the spacetime information
        spacetime: Optional[Dict[str, List[Tree]]] = None
        if output.root_name() in self.mapping.get_spacetime().keys():
            spacetime = self.mapping.get_spacetime()[output.root_name()]

        if spacetime is not None:
            # Build the spacetime object
            self.spacetime = SpaceTime(
                spacetime, self.partitioning, output.root_name())

    def apply_all_partitioning(self, tensor: Tensor) -> None:
        """
        Partition the tensor according to the partitioning given in
        add_einsum()
        """
        # Make sure that the program is configured
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        new_ranks = self.partitioning.partition_tensor(
            tensor, tensor.get_ranks(), True)
        tensor.update_ranks(new_ranks)

    def apply_partitioning(self, tensor: Tensor, rank: str) -> None:
        """
        Partition the tensor according to the partitioning given in
        add_einsum() for the given rank
        """

        # Make sure that the program is configured
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        new_ranks = self.partitioning.partition_tensor(tensor, [rank], False)
        tensor.update_ranks(new_ranks)

    def get_einsum(self) -> Tree:
        """
        Get the parse tree representation of the einsum
        """
        # Make sure that the program is configured
        if self.equation is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.equation

    def get_einsum_ind(self) -> int:
        """
        Get the index of this einsum
        """
        # Make sure that the program is configured
        if self.einsum_ind is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.einsum_ind

    def get_coord_math(self) -> CoordMath:
        """
        Get the CoordMath intermediate representation
        """
        # Make sure that the program is configured
        if self.coord_math is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.coord_math

    def get_loop_order(self) -> LoopOrder:
        """
        Get the LoopOrder intermediate representation
        """
        # Make sure that the program is configured
        if self.loop_order is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.loop_order

    def get_output(self) -> Tensor:
        """
        Get the output tensor used for this kernel
        """
        # Make sure that the program is configured
        if self.loop_order is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.es_tensors[0]

    def get_partitioning(self) -> Partitioning:
        """
        Get the partitioning information for the current Einsum
        """
        # Make sure that the program is configured
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.partitioning

    def get_spacetime(self) -> Optional[SpaceTime]:
        """
        Get the spacetime information for this kernel, should it exist
        """
        # Make sure the program is configured
        # Note: we have to check another field, since it is possible for spacetime
        # to be empty even in a configured program
        if self.loop_order is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.spacetime

    def get_tensor(self, root_name: str) -> Tensor:
        """
        Get a tensor currently being used from its root name
        """
        # Make sure that the program is configured
        if not self.es_tensors:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        for tensor in self.es_tensors:
            if tensor.root_name() == root_name:
                return tensor

        raise ValueError("Unknown tensor " + root_name)

    def get_tensors(self) -> List[Tensor]:
        """
        Get the tensors used in an einsum
        """
        # Make sure that the program is configured
        if not self.es_tensors:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.es_tensors

    def reset(self) -> None:
        """
        Unconfigure the program and corresponding tensors
        """
        for tensor in self.tensors.values():
            tensor.reset()

        self.equation = None
        self.es_tensors = []
        self.loop_order = None
        self.partitioning = None
        self.spacetime = None

    def __add_tranks(self, tensor: Tensor, tensor_tree: Tree) -> None:
        """
        Add the tranks of a tensor to the coord math
        """
        # Note: this should always be called through add_einsum(), so we should
        # never encounter this problem
        if self.coord_math is None:
            raise ValueError("Something is wrong...")  # pragma: no cover

        tranks = next(tensor_tree.find_data("tranks"))
        self.coord_math.add(self.decl_tensors[tensor.root_name()], tranks)

    def __get_tensor(self, tensor: Tree) -> Tensor:
        """
        Given a parse tree, get the appropriate tensor
        """
        name = ParseUtils.next_str(tensor)
        if name not in self.tensors.keys():
            raise ValueError("Undeclared tensor: " + name)

        return self.tensors[name]

    def __all_ranks(self) -> Set[str]:
        """
        Get the set of all ranks
        """
        if self.equation is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        ranks = set()
        for tensor in self.es_tensors:
            ranks.update(tensor.get_ranks())

        for sranks in self.equation.find_data("sranks"):
            for rank in sranks.children:
                ranks.add(str(rank))

        return ranks