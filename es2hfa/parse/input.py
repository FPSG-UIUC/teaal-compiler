"""
Parse the input YAML file
"""

from typing import List, Optional

from lark.tree import Tree

from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.tensor import TensorParser
from es2hfa.parse.yaml import YamlParser


class Input:
    """
    Parse the input YAML file into the input to the compiler
    """

    def __init__(self, filename: str) -> None:
        """
        Read the YAML file input
        """
        yaml = YamlParser.parse(filename)

        # Parse the Einsums
        self.declaration = [TensorParser.parse(
            tensor) for tensor in yaml["einsum"]["declaration"]]

        self.exprs = [EinsumParser.parse(expr)
                      for expr in yaml["einsum"]["expressions"]]

        # If a mapping exists, parse the mapping
        if "mapping" in yaml.keys():
            mapping = yaml["mapping"]
            if "rank-order" in mapping.keys():
                self.rank_orders = [TensorParser.parse(
                    tensor) for tensor in mapping["rank-order"]]
            else:
                self.rank_orders = []

            if "loop-order" in mapping.keys():
                self.loop_orders = mapping["loop-order"]
            else:
                self.loop_orders = None

        else:
            self.rank_orders = []
            self.loop_orders = None

    def get_declaration(self) -> List[Tree]:
        """
        Get the declaration
        """
        return self.declaration

    def get_expressions(self) -> List[Tree]:
        """
        Get the expressions (Einsums)
        """
        return self.exprs

    def get_loop_order(self, output: str) -> Optional[List[str]]:
        """
        Get the loop order for a specified output tensor
        """
        if self.loop_orders:
            if output in self.loop_orders.keys():
                return self.loop_orders[output]
            else:
                return None
        else:
            return None

    def get_rank_orders(self) -> List[Tree]:
        """
        Get any rank orders specified
        """
        return self.rank_orders
