"""
Parse the input YAML file
"""

from typing import Dict, List, Optional

from lark.tree import Tree

from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.partitioning import PartitioningParser
from es2hfa.parse.yaml import YamlParser


class Input:
    """
    Parse the input YAML file into the input to the compiler
    """

    def __init__(self, yaml: dict) -> None:
        """
        Read the YAML file input
        """
        # Parse the Einsums
        self.declaration = yaml["einsum"]["declaration"]

        self.exprs = [EinsumParser.parse(expr)
                      for expr in yaml["einsum"]["expressions"]]

        # If a mapping exists, parse the mapping
        rank_orders = None
        loop_orders = None
        partitioning: Optional[Dict[str, Dict[str, List[Tree]]]] = None
        if "mapping" in yaml.keys():
            mapping = yaml["mapping"]

            if "rank-order" in mapping.keys():
                rank_orders = mapping["rank-order"]

            if "loop-order" in mapping.keys():
                loop_orders = mapping["loop-order"]

            if "partitioning" in mapping.keys():
                partitioning = {}
                for tensor, inds in mapping["partitioning"].items():
                    partitioning[tensor] = {}

                    if inds is None:
                        continue

                    for ind, parts in inds.items():
                        partitioning[tensor][ind] = []
                        for part in parts:
                            partitioning[tensor][ind].append(
                                PartitioningParser.parse(part))

        if rank_orders is None:
            self.rank_orders = {}
        else:
            self.rank_orders = rank_orders

        if loop_orders is None:
            self.loop_orders = {}
        else:
            self.loop_orders = loop_orders

        if partitioning is None:
            self.partitioning = {}
        else:
            self.partitioning = partitioning

    @classmethod
    def from_file(cls, filename: str) -> "Input":
        """
        Construct a new Input from a YAML file
        """
        return cls(YamlParser.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Input":
        """
        Construct a new Input from a string in the YAML format
        """
        return cls(YamlParser.parse_str(string))

    def get_declaration(self) -> Dict[str, List[str]]:
        """
        Get the declaration
        """
        return self.declaration

    def get_expressions(self) -> List[Tree]:
        """
        Get the expressions (Einsums)
        """
        return self.exprs

    def get_loop_orders(self) -> Dict[str, List[str]]:
        """
        Get the dictionary from output tensors to loop orders
        """
        return self.loop_orders

    def get_partitioning(self) -> Dict[str, Dict[str, List[Tree]]]:
        """
        Get a dictionary from output tensors to a dictionary of index variables
        to partitioning information
        """
        return self.partitioning

    def get_rank_orders(self) -> Dict[str, List[str]]:
        """
        Get any rank orders specified
        """
        return self.rank_orders

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Inputs
        """
        if isinstance(other, type(self)):
            return self.declaration == other.declaration and \
                self.exprs == other.exprs and \
                self.loop_orders == other.loop_orders and \
                self.partitioning == other.partitioning and \
                self.rank_orders == other.rank_orders
        return False
