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

Parsing the yaml input file
"""
# Known issue with ruamel and mypy (https://github.com/python/mypy/issues/7276)
from ruamel.yaml import YAML  # type: ignore


class YamlParser:
    """
    Parser for the input YAML text
    """

    @staticmethod
    def parse_str(string: str) -> dict:
        """
        Parse a string in the YAML format into the corresponding dictionary
        """
        yaml = YAML(typ='safe', pure=True)
        return yaml.load(string)

    @staticmethod
    def parse_file(input_file: str) -> dict:
        """
        Parse a YAML file into the corresponding dictionary
        """
        with open(input_file, 'r') as stream:
            yaml = YAML(typ='safe', pure=True)
            data_loaded = yaml.load(stream)
        return data_loaded
