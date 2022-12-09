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

Representation of a graph node
"""

import abc
from typing import Any, Iterable


class Node(metaclass=abc.ABCMeta):
    """
    Graph node interface
    """

    def __eq__(self, other: object) -> bool:
        """
        The == operator for nodes

        """
        if isinstance(other, type(self)):
            return self.__key() == other.__key()
        return False

    def __hash__(self) -> int:
        """
        Hash the node (needed to insert it into the graph)
        """
        return hash(repr(self))

    def __key(self) -> Iterable[Any]:
        """
        A tuple of all fields of a node
        """
        return ()

    def __repr__(self) -> str:
        """
        A string representation of the node for hashing
        """
        strs = [key if isinstance(key, str) else repr(key)
                for key in self.__key()]
        return "(" + type(self).__name__ + ", " + ", ".join(strs) + ")"
