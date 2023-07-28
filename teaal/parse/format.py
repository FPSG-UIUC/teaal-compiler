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

Parse the input YAML for the format
"""

from typing import Optional

from teaal.parse.yaml import YamlParser


class Format:
    """
    Parse the input YAML for the format
    """

    def __init__(self, yaml: Optional[dict]) -> None:
        """
        Read the YAML input
        """
        # If there is an format specification, parse it
        if yaml is None or "format" not in yaml.keys():
            self.yaml = {}
            return

        self.yaml = yaml["format"]

        for tensor, formats in self.yaml.items():
            for format_, spec in formats.items():
                if "rank-order" not in spec.keys():
                    raise ValueError("Rank order not specified for tensor " + tensor + " in format " + format_)

    @classmethod
    def from_file(cls, filename: str) -> "Format":
        """
        Construct a new Format from a YAML file
        """
        return cls(YamlParser.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Format":
        """
        Construct a new Format from a string in the YAML format
        """
        return cls(YamlParser.parse_str(string))

    def get_spec(self, tensor: str) -> dict:
        """
        Get the specification for a particular tensor
        """
        if tensor not in self.yaml.keys():
            return {}

        return self.yaml[tensor]
