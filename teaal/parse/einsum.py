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

Parse the einsum of the input YAML
"""

from lark.tree import Tree
from typing import Dict, List

from teaal.parse.equation import EquationParser
from teaal.parse.yaml import YamlParser


class Einsum:
    """
    Parse the einsum of the input YAML for the compiler
    """

    def __init__(self, yaml: dict) -> None:
        """
        Read the YAML input
        """
        # Parse the Einsums
        self.declaration = yaml["einsum"]["declaration"]

        self.exprs = [EquationParser.parse(expr)
                      for expr in yaml["einsum"]["expressions"]]

    @classmethod
    def from_file(cls, filename: str) -> "Einsum":
        """
        Construct a new Einsum from a YAML file
        """
        return cls(YamlParser.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Einsum":
        """
        Construct a new Einsum from a string in the YAML format
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

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Einsums
        """
        if isinstance(other, type(self)):
            return self.declaration == other.declaration and \
                self.exprs == other.exprs
        return False
