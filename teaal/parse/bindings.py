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
        self.configs = {}
        self.prefixes = {}
        self.energy = {}
        if yaml is None or "bindings" not in yaml.keys():
            return

        for einsum in yaml["bindings"]:
            self.components[einsum] = {}

            configured = False
            for binding in yaml["bindings"][einsum]:
                if "config" in binding:
                    self.configs[einsum] = binding["config"]
                    self.prefixes[einsum] = binding["prefix"]

                    if "energy" in binding:
                        self.energy[einsum] = binding["energy"]
                    else:
                        self.energy[einsum] = False

                    configured = True

                else:
                    self.components[einsum][binding["component"]
                                            ] = binding["bindings"]

            if not configured:
                raise ValueError(
                    "Accelerator config and prefix missing for Einsum " + einsum)

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

    def get_component(self, name: str) -> Dict[str, List[dict]]:
        """
        Get the binding information for a component
        """
        info = {}

        for einsum in self.components:
            if name in self.components[einsum].keys():
                info[einsum] = self.components[einsum][name]

        return info

    def get_bindings(self) -> Dict[str, Dict[str, List[dict]]]:
        """
        Get the binding information for all components
        """
        return self.components

    def get_config(self, einsum: str) -> str:
        """
        Get the hardware configuration for a given Einsum
        """
        return self.configs[einsum]

    def get_energy(self, einsum: str) -> bool:
        """
        Get whether or not energy should be tracked for the given Einsum
        """
        return self.energy[einsum]

    def get_prefix(self, einsum: str) -> str:
        """
        Get the metrics prefix for the given Einsum
        """
        return self.prefixes[einsum]
