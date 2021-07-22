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
from typing import cast, Dict, List, Optional, Union

from es2hfa.ir.loop_order import LoopOrder
from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.spacetime import SpaceTime
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.parse.utils import ParseUtils


class Program:
    """
    Store tensor metadata and configure the metadata for specific einsums
    """

    def __init__(self, einsum: Einsum, mapping: Mapping) -> None:
        """
        Construct the metadata for tensors
        """
        self.einsum = einsum
        self.mapping = mapping

        # Get all tensors
        self.tensors = {}
        declaration = self.einsum.get_declaration()
        for ten_name in declaration:
            tensor = Tensor(ten_name, declaration[ten_name])
            self.tensors[tensor.root_name()] = tensor

        # Replace the tensors whose rank order is specified
        rank_orders = self.mapping.get_rank_orders()
        for ord_name in rank_orders:
            tensor = Tensor(ord_name, rank_orders[ord_name])
            if tensor.root_name() not in self.tensors.keys():
                raise ValueError("Undeclared tensor: " + tensor.root_name())

            self.tensors[tensor.root_name()] = tensor

        self.equation: Optional[Tree] = None
        self.es_tensors: List[Tensor] = []
        self.loop_order: Optional[LoopOrder] = None
        self.partitioning: Optional[Partitioning] = None
        self.spacetime: Optional[SpaceTime] = None

    def add_einsum(self, i: int) -> None:
        """
        Configure the program for the i'th Einsum
        """
        self.equation = self.einsum.get_expressions()[i]

        # Build the list of tensors, starting with the output tensor
        self.es_tensors = []
        output = self.__get_tensor(next(self.equation.find_data("output")))
        output.set_is_output(True)
        self.es_tensors.append(output)

        # Add the rest of the tensors
        for tensor_tree in self.equation.find_data("tensor"):
            self.es_tensors.append(self.__get_tensor(tensor_tree))

        # Create the loop_order object
        self.loop_order = LoopOrder(self.equation, output)

        # Store the partitioning information
        partitioning = self.mapping.get_partitioning()
        inds = self.loop_order.get_unpartitioned_inds()
        if output.root_name() in partitioning.keys():
            self.partitioning = Partitioning(
                partitioning[output.root_name()], inds)
        else:
            self.partitioning = Partitioning({}, inds)

        # Store the loop order
        loop_orders = self.mapping.get_loop_orders()
        opt_loop_order: Optional[List[str]] = None
        if output.root_name() in loop_orders.keys():
            opt_loop_order = loop_orders[output.root_name()]

        self.loop_order.add_loop_order(opt_loop_order, self.partitioning)

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

        tensor.partition(self.partitioning.get_all_parts())

    def apply_dyn_partitioning(self, tensor: Tensor, ind: str) -> None:
        """
        Partition the tensor according to the dynamic partitioning given in
        add_einsum() for the given rank
        """
        # Make sure that the program is configured
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        tensor.partition({ind: self.partitioning.get_dyn_parts()[ind]})

    def apply_loop_order(self, tensor: Tensor) -> None:
        """
        Swizzle the given tensor with the loop order
        """
        # Make sure that the program is configured
        if self.loop_order is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        tensor.swizzle(
            cast(List[Optional[str]], self.loop_order.get_loop_order()))

    def apply_static_partitioning(self, tensor: Tensor) -> None:
        """
        Partition the tensor according to the static partitioning given in
        add_einsum()
        """
        # Make sure that the program is configured
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        tensor.partition(self.partitioning.get_static_parts())

    def get_einsum(self) -> Tree:
        """
        Get the parse tree representation of the einsum
        """
        # Make sure that the program is configured
        if self.equation is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.equation

    def get_loop_order(self) -> List[str]:
        """
        Get the loop order used for this kernel
        """
        # Make sure that the program is configured
        if self.loop_order is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        return self.loop_order.get_loop_order()

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
        for tensor in self.es_tensors:
            tensor.reset()

        self.es_tensors = []
        self.loop_order = None
        self.partitioning = None
        self.spacetime = None

    def start_partitioning(self, ind: str) -> None:
        """
        Start partitioning the dimension given
        """
        # Make sure that the program is configured
        if self.loop_order is None or self.partitioning is None:
            raise ValueError(
                "Unconfigured program. Make sure to first call add_einsum()")

        self.partitioning.partition_dim(ind)
        self.loop_order.update_loop_order()

    def __get_tensor(self, tensor: Tree) -> Tensor:
        """
        Given a parse tree, get the appropriate tensor
        """
        name = ParseUtils.next_str(tensor)
        if name not in self.tensors.keys():
            raise ValueError("Undeclared tensor: " + name)

        return self.tensors[name]
