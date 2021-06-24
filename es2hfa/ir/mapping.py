"""
Representation of tensor metadata
"""

from typing import cast, Dict, Generator, List, Optional

from lark.tree import Tree

from es2hfa.ir.tensor import Tensor
from es2hfa.parse.input import Input


class Mapping:
    """
    Store tensor metadata and configure the metadaa for specific einsums
    """

    def __init__(self, input_: Input) -> None:
        """
        Construct the metadata for tensors
        """
        self.input = input_

        # Get all tensors
        self.tensors = {}
        declaration = self.input.get_declaration()
        for ten_name in declaration:
            tensor = Tensor(ten_name, declaration[ten_name])
            self.tensors[tensor.root_name()] = tensor

        # Replace the tensors whose rank order is specified
        rank_orders = self.input.get_rank_orders()
        for ord_name in rank_orders:
            tensor = Tensor(ord_name, rank_orders[ord_name])
            if tensor.root_name() not in self.tensors.keys():
                raise ValueError("Undeclared tensor: " + tensor.root_name())

            self.tensors[tensor.root_name()] = tensor

        self.es_tensors: List[Tensor] = []
        self.loop_order: Optional[List[str]] = None
        self.partitioning: Optional[Dict[str, List[Tree]]] = None

    def add_einsum(self, i: int) -> None:
        """
        Configure the mapping for the ith Einsum
        """
        einsum = self.input.get_expressions()[i]

        # Build the list of tensors, starting with the output tensor
        self.es_tensors = []
        output = self.__get_tensor(next(einsum.find_data("output")))
        output.set_is_output(True)
        self.es_tensors.append(output)

        # Add the rest of the tensors
        for tensor_tree in einsum.find_data("tensor"):
            self.es_tensors.append(self.__get_tensor(tensor_tree))

        # Store the partitioning information
        partitioning = self.input.get_partitioning()
        if output.root_name() in partitioning.keys():
            self.partitioning = partitioning[output.root_name()]
        else:
            self.partitioning = {}

        # Store the loop order
        loop_orders = self.input.get_loop_orders()
        if output.root_name() in loop_orders.keys():
            self.loop_order = loop_orders[output.root_name()]

        if self.loop_order is None:
            self.__default_loop_order(einsum)

    def apply_loop_order(self, tensor: Tensor) -> None:
        """
        Swizzle the given tensor with the loop order
        """
        # Make sure that the mapping is configured
        if self.loop_order is None:
            raise ValueError(
                "Unconfigured mapping. Make sure to first call add_einsum()")

        tensor.swizzle(cast(List[Optional[str]], self.loop_order))

    def apply_partitioning(self, tensor: Tensor) -> None:
        """
        Partition the tensor according to the schedule given in add_einsum()
        """
        # Make sure that the mapping is configured
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured mapping. Make sure to first call add_einsum()")

        tensor.partition(self.partitioning)

    def get_loop_order(self) -> List[str]:
        """
        Get the loop order used for this kernel
        """
        # Make sure that the mapping is configured
        if self.loop_order is None:
            raise ValueError(
                "Unconfigured mapping. Make sure to first call add_einsum()")

        return self.loop_order

    def get_output(self) -> Tensor:
        """
        Get the output tensor used for this kernel
        """
        # Make sure that the mapping is configured
        if self.loop_order is None:
            raise ValueError(
                "Unconfigured mapping. Make sure to first call add_einsum()")

        return self.es_tensors[0]

    def get_partitioning(self, tensor: Tensor) -> Dict[str, List[Tree]]:
        """
        Get all of the partitioning information relevant for a given tensor
        """
        # Make sure that the mapping is configured
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured mapping. Make sure to first call add_einsum()")

        return {ind: parts for ind, parts in self.partitioning.items()
                if ind in tensor.get_inds()}

    def get_tensors(self) -> List[Tensor]:
        """
        Get the tensors used in an einsum
        """
        # Make sure that the mapping is configured
        if not self.es_tensors:
            raise ValueError(
                "Unconfigured mapping. Make sure to first call add_einsum()")

        return self.es_tensors

    def reset(self) -> None:
        """
        Unconfigure the mapping and corresponding tensors
        """
        for tensor in self.es_tensors:
            tensor.reset()

        self.es_tensors = []
        self.loop_order = None
        self.partitioning = None

    def __default_loop_order(self, einsum: Tree) -> None:
        """
        Compute the default loop order
        """
        if self.partitioning is None:
            raise ValueError("Must configure partitioning before loop order")

        self.loop_order = self.es_tensors[0].get_inds().copy()

        for sum_ in einsum.find_data("sum"):
            self.loop_order += list(next(sum_.find_data("sinds")
                                         ).scan_values(lambda _: True))

        for ind, parts in self.partitioning.items():
            # Remove the old index
            i = self.loop_order.index(ind)
            self.loop_order.pop(i)

            # Insert the new indices
            new_inds = [ind + str(j) for j in range(len(parts) + 1)]
            for new_ind in new_inds:
                self.loop_order.insert(i, new_ind)

    def __get_tensor(self, tensor: Tree) -> Tensor:
        """
        Given a parse tree, get the appropriate tensor
        """
        name = next(cast(Generator, tensor.scan_values(lambda _: True)))
        if name not in self.tensors.keys():
            raise ValueError("Undeclared tensor: " + name)

        return self.tensors[name]
