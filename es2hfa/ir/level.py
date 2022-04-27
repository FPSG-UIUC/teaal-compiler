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

Representation of a level of the architecture hierarchy
"""

from typing import Any, Iterable, List, Union

from es2hfa.ir.component import Component


class Level:
    """
    Representation of a level of the architecture hierarchy
    """

    def __init__(
            self,
            name: str,
            num: int,
            attrs: dict,
            local: List[Component],
            subtrees: List["Level"]):
        """
        Construct a level
        """
        self.name = name
        self.num = num
        self.attrs = attrs
        self.local = local
        self.subtrees = subtrees

    def get_attr(self, attr: str) -> Union[None, int, str]:
        """
        Get an attribute
        """
        if attr not in self.attrs.keys():
            return None

        return self.attrs[attr]

    def get_local(self) -> List[Component]:
        """
        Get the local components
        """
        return self.local

    def get_name(self) -> str:
        """
        Get the name of this level
        """
        return self.name

    def get_num(self) -> int:
        """
        Return the number of copies of this level
        """
        return self.num

    def get_subtrees(self) -> List["Level"]:
        """
        Get the subtrees under this level
        """
        return self.subtrees

    def __eq__(self, other: object) -> bool:
        """
        The == operator for levels

        """
        if isinstance(other, type(self)):
            return self.__key() == other.__key()
        return False

    def __key(self) -> Iterable[Any]:
        """
        A tuple of all fields of a level
        """
        return (self.name, self.num, self.attrs, self.local, self.subtrees)

    def __repr__(self) -> str:
        """
        A string representation of the level
        """
        strs = [key if isinstance(key, str) else repr(key)
                for key in self.__key()]
        return "(" + type(self).__name__ + ", " + ", ".join(strs) + ")"
