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

Parse tree utilities
"""

from lark.tree import Tree
from typing import cast, Generator


class ParseUtils:
    """
    Class to wrap parse tree utilities
    """
    @staticmethod
    def find_int(tree: Tree, data: str) -> int:
        """
        Find the next subtree with the given data and get the next integer
        token from that subtree
        """
        return ParseUtils.next_int(next(tree.find_data(data)))

    @staticmethod
    def find_str(tree: Tree, data: str) -> str:
        """
        Find the next subtree with the given data and get the next string token
        from that subtree
        """
        return ParseUtils.next_str(next(tree.find_data(data)))

    @staticmethod
    def next_int(tree: Tree) -> int:
        """
        Get the next token in the tree
        """
        return int(next(cast(Generator, tree.scan_values(lambda _: True))))

    @staticmethod
    def next_str(tree: Tree) -> str:
        """
        Get the next token in the tree
        """
        return str(next(cast(Generator, tree.scan_values(lambda _: True))))
