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

Parse the input YAML for the mapping
"""

from lark.tree import Tree
from typing import Dict, List, Optional, Union

from teaal.parse.spacetime import SpaceTimeParser
from teaal.parse.partitioning import PartitioningParser
from teaal.parse.yaml import YamlParser


class Mapping:
    """
    Parse the input YAML for the mapping
    """

    def __init__(self, yaml: Optional[dict]) -> None:
        """
        Read the YAML input
        """
        loop_orders = None
        partitioning: Optional[Dict[str, Dict[Tree, List[Tree]]]] = None
        rank_orders = None
        # Use type dict, since the dictionary is very heterogeneous
        spacetime: Optional[dict] = None

        if yaml is not None and "mapping" in yaml.keys() and \
                yaml["mapping"] is not None:
            mapping = yaml["mapping"]

            if "loop-order" in mapping.keys():
                loop_orders = mapping["loop-order"]

            if "partitioning" in mapping.keys():
                partitioning = {}
                for tensor, ranks in mapping["partitioning"].items():
                    partitioning[tensor] = {}

                    if ranks is None:
                        continue

                    for ranks_str, parts in ranks.items():
                        ranks_tree = PartitioningParser.parse_ranks(ranks_str)
                        partitioning[tensor][ranks_tree] = []
                        for part in parts:
                            partitioning[tensor][ranks_tree].append(
                                PartitioningParser.parse_partitioning(part))

            if "rank-order" in mapping.keys():
                rank_orders = mapping["rank-order"]

            if "spacetime" in mapping.keys(
            ) and mapping["spacetime"] is not None:
                spacetime = {}
                for tensor, info in mapping["spacetime"].items():
                    spacetime[tensor] = {}

                    # Parse the space and time stamps
                    for stamp in ["space", "time"]:
                        spacetime[tensor][stamp] = []
                        for rank in info[stamp]:
                            spacetime[tensor][stamp].append(
                                SpaceTimeParser.parse(rank))

                    # Store any other optimizations
                    if "opt" in info.keys():
                        spacetime[tensor]["opt"] = info["opt"]

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

        if spacetime is None:
            self.spacetime = {}
        else:
            self.spacetime = spacetime

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

    def get_loop_orders(self) -> Dict[str, List[str]]:
        """
        Get the dictionary from output tensors to loop orders
        """
        return self.loop_orders

    def get_partitioning(self) -> Dict[str, Dict[Tree, List[Tree]]]:
        """
        Get a dictionary from output tensors to a dictionary of rank variables
        to partitioning information
        """
        return self.partitioning

    def get_rank_orders(self) -> Dict[str, List[str]]:
        """
        Get any rank orders specified
        """
        return self.rank_orders

    def get_spacetime(self) -> dict:
        """
        Get the spacetime information
        """
        return self.spacetime

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Mappings
        """
        if isinstance(other, type(self)):
            return self.spacetime == other.spacetime and \
                self.loop_orders == other.loop_orders and \
                self.partitioning == other.partitioning and \
                self.rank_orders == other.rank_orders
        return False
