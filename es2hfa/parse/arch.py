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

Parse the input YAML for the architecture
"""

from lark.tree import Tree
from typing import Optional

from es2hfa.parse.level import LevelParser
from es2hfa.parse.yaml import YamlParser


class Architecture:
    """
    Parse the input YAML for the architecture
    """

    def __init__(self, yaml: Optional[dict]) -> None:
        """
        Read the YAML input
        """
        # If there is an architecture specification, parse it
        self.yaml = yaml
        if self.yaml is None:
            return

        # We need to parse the tree names, check for errors, and add omitted
        # attributes
        if not isinstance(self.yaml["architecture"], dict):
            raise ValueError("Bad architecture spec: " + str(self.yaml))

        subtrees = self.yaml["architecture"]["subtree"].copy()

        while subtrees:
            tree = subtrees.pop()

            if "name" not in tree.keys():
                raise ValueError("Unnamed subtree: " + repr(tree))

            name_tree = LevelParser.parse(tree["name"])

            if name_tree.data == "single":
                tree["name"] = str(name_tree.children[0])
                tree["num"] = 1

            elif name_tree.data == "multiple":
                tree["name"] = str(name_tree.children[0])

                num = name_tree.children[1]
                if isinstance(num, Tree):
                    # This error should be caught by the LevelParser
                    raise ValueError(
                        "Unknown num: " + repr(num))  # pragma: no cover

                tree["num"] = int(num) + 1

            else:
                # This error should be caught by the LevelParser
                raise ValueError(
                    "Unknown level name: " +
                    repr(name_tree))  # pragma: no cover

            if "attributes" not in tree.keys():
                tree["attributes"] = {}

            if "local" not in tree.keys():
                tree["local"] = []

            for local in tree["local"]:
                if "name" not in local.keys():
                    raise ValueError("Unnamed local: " + repr(local))

                if "class" not in local.keys():
                    raise ValueError("Unclassed local: " + repr(local))

                if "attributes" not in local.keys():
                    local["attributes"] = {}

            if "subtree" not in tree.keys():
                tree["subtree"] = []

            subtrees.extend(tree["subtree"])

    @classmethod
    def from_file(cls, filename: str) -> "Architecture":
        """
        Construct a new Architecture from a YAML file
        """
        return cls(YamlParser.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Architecture":
        """
        Construct a new Architecture from a string in the YAML format
        """
        return cls(YamlParser.parse_str(string))

    def get_spec(self) -> Optional[dict]:
        """
        Get the cleaned specification
        """
        return self.yaml
