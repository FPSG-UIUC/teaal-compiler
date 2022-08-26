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

Parse the input YAML for the bindings
"""

from typing import List, Optional

from teaal.parse.yaml import YamlParser


class Bindings:
    """
    Parse the input YAML for the bindings
    """

    def __init__(self, yaml: Optional[dict]) -> None:
        """
        Read the YAML input
        """

        self.components = {}
        if yaml is None or "bindings" not in yaml.keys():
            return

        for binding in yaml["bindings"]:
            self.components[binding["name"]] = binding["bindings"]

    @classmethod
    def from_file(cls, filename: str) -> "Bindings":
        """
        Construct a new Bindings from a YAML file
        """
        return cls(YamlParser.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Bindings":
        """
        Construct a new Bindings from a string in the YAML format
        """
        return cls(YamlParser.parse_str(string))

    def get(self, name) -> List[dict]:
        """
        Get the binding information for a component
        """
        if name not in self.components.keys():
            return []

        return self.components[name]
