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

Parse the input YAML
"""

from lark.tree import Tree
from typing import Dict, List, Optional, Union

from es2hfa.parse.display import DisplayParser
from es2hfa.parse.partitioning import PartitioningParser
from es2hfa.parse.yaml import YamlParser


class Mapping:
    """
    Parse the input YAML for the compiler
    """

    def __init__(self, yaml: dict) -> None:
        """
        Read the YAML input
        """
        # If a mapping exists, parse the mapping
        display: Optional[Dict[str, Dict[str, List[Tree]]]] = None
        loop_orders = None
        partitioning: Optional[Dict[str, Dict[str, List[Tree]]]] = None
        rank_orders = None

        if "mapping" in yaml.keys():
            mapping = yaml["mapping"]

            if "display" in mapping.keys() and mapping["display"] is not None:
                display = {}
                for tensor, info in mapping["display"].items():
                    display[tensor] = {}

                    for stamp, inds in info.items():
                        display[tensor][stamp] = []
                        for ind in inds:
                            display[tensor][stamp].append(
                                DisplayParser.parse(ind))

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

            if "rank-order" in mapping.keys():
                rank_orders = mapping["rank-order"]

        if display is None:
            self.display = {}
        else:
            self.display = display

        if loop_orders is None:
            self.loop_orders = {}
        else:
            self.loop_orders = loop_orders

        if partitioning is None:
            self.partitioning = {}
        else:
            self.partitioning = partitioning

        if rank_orders is None:
            self.rank_orders = {}
        else:
            self.rank_orders = rank_orders

    @classmethod
    def from_file(cls, filename: str) -> "Mapping":
        """
        Construct a new Mapping from a YAML file
        """
        return cls(YamlParser.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Mapping":
        """
        Construct a new Mapping from a string in the YAML format
        """
        return cls(YamlParser.parse_str(string))

    def get_display(self) -> Dict[str, Dict[str, List[Tree]]]:
        """
        Get the display information
        """
        return self.display

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
        The == operator for Mappings
        """
        if isinstance(other, type(self)):
            return self.display == other.display and \
                self.loop_orders == other.loop_orders and \
                self.partitioning == other.partitioning and \
                self.rank_orders == other.rank_orders
        return False
