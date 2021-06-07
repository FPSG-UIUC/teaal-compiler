"""
Representation of tensor metadata
"""

from typing import cast, Dict, Generator, List, Optional

from lark.tree import Tree

from es2hfa.ir.tensor import Tensor


class Mapping:
    """
    Store tensor metadata and configure the metadaa for specific einsums
    """

    def __init__(
            self,
            declaration: List[Tree],
            rank_orders: List[Tree]) -> None:
        """
        Construct the metadata for tensors
        """
        # Get all tensors
        self.tensors = {}
        for declared in declaration:
            tensor = Tensor(declared)
            self.tensors[tensor.root_name()] = tensor

        # Replace the tensors whose rank order is specified
        for ordered in rank_orders:
            tensor = Tensor(ordered)
            if tensor.root_name() not in self.tensors.keys():
                raise ValueError("Undeclared tensor: " + tensor.root_name())

            self.tensors[tensor.root_name()] = tensor

        self.es_tensors: List[Tensor] = []
        self.loop_order: Optional[List[str]] = None

    def add_einsum(self, einsum: Tree,
                   loop_orders: Dict[str, List[str]]) -> None:
        """
        Configure the mapping for a particular Einsum
        """
        # Make sure we are starting with the full einsum
        if einsum.data != "einsum":
            raise ValueError("Input parse tree must be an einsum")

        # Build the list of tensors, starting with the output tensor
        self.es_tensors = []
        output = self.__get_tensor(next(einsum.find_data("output")))
        output.set_is_output(True)
        self.es_tensors.append(output)

        # Add the rest of the tensors
        for tensor_tree in einsum.find_data("tensor"):
            self.es_tensors.append(self.__get_tensor(tensor_tree))

        # Store the loop order
        if output.root_name() in loop_orders.keys():
            self.loop_order = loop_orders[output.root_name()]
        else:
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

    def get_loop_order(self) -> List[str]:
        """
        Get the loop order used for this kernel
        """
        # Make sure that the mapping is configured
        if self.loop_order is None:
            raise ValueError(
                "Unconfigured mapping. Make sure to first call add_einsum()")

        return self.loop_order

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

    def __default_loop_order(self, einsum: Tree) -> None:
        """
        Compute the default loop order
        """
        self.loop_order = self.es_tensors[0].get_inds().copy()

        for sum_ in einsum.find_data("sum"):
            self.loop_order += list(next(sum_.find_data("sinds")
                                         ).scan_values(lambda _: True))

    def __get_tensor(self, tensor: Tree) -> Tensor:
        """
        Given a parse tree, get the appropriate tensor
        """
        name = next(cast(Generator, tensor.scan_values(lambda _: True)))
        if name not in self.tensors.keys():
            raise ValueError("Undeclared tensor: " + name)

        return self.tensors[name]
