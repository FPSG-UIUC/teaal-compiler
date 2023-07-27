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

from typing import Dict, List, Optional

from teaal.parse.yaml import YamlParser


class Bindings:
    """
    Parse the input YAML for the bindings
    """

    def __init__(self, yaml: Optional[dict]) -> None:
        """
        Read the YAML input
        """

        self.components: Dict[str, Dict[str, List[dict]]] = {}
        if yaml is None or "bindings" not in yaml.keys():
            return

        self.configs = {}
        for einsum in yaml["bindings"]:
            self.components[einsum] = {}

            for binding in yaml["bindings"][einsum]:
                if "config" in binding:
                    self.configs[einsum] = binding["config"]
                else:
                    self.components[einsum][binding["component"]
                                            ] = binding["bindings"]

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

    def get(self, name: str) -> Dict[str, List[dict]]:
        """
        Get the binding information for a component
        """
        info = {}

        for einsum in self.components:
            if name in self.components[einsum].keys():
                info[einsum] = self.components[einsum][name]

        return info

    def get_config(self, einsum: str) -> str:
        """
        Get the hardware configuration for a given Einsum
        """
        return self.configs[einsum]
