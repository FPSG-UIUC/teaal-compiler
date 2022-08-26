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

HFA AST base classes
"""

import abc


class Argument(metaclass=abc.ABCMeta):
    """
    Argument interface
    """

    @abc.abstractmethod
    def gen(self) -> str:
        """
        Generate the HFA code for this Argument
        """
        raise NotImplementedError  # pragma: no cover


class Assignable(metaclass=abc.ABCMeta):
    """
    Assignable interface
    """

    @abc.abstractmethod
    def gen(self) -> str:
        """
        Generate the HFA code for this Assignable
        """
        raise NotImplementedError  # pragma: no cover


class Expression(metaclass=abc.ABCMeta):
    """
    Expression interface
    """

    @abc.abstractmethod
    def gen(self) -> str:
        """
        Generate the HFA code for this Expression
        """
        raise NotImplementedError  # pragma: no cover


class Operator(metaclass=abc.ABCMeta):
    """
    Operator interface
    """

    @abc.abstractmethod
    def gen(self) -> str:
        """
        Generate the HFA code for this Operator
        """
        raise NotImplementedError  # pragma: no cover


class Payload(metaclass=abc.ABCMeta):
    """
    Payload interface
    """

    @abc.abstractmethod
    def gen(self, parens: bool) -> str:
        """
        Generate the HFA code for this Payload
        """
        raise NotImplementedError  # pragma: no cover


class Statement(metaclass=abc.ABCMeta):
    """
    Statement interface
    """

    @abc.abstractmethod
    def gen(self, depth: int) -> str:
        """
        Generate the HFA code for this Statement
        """
        raise NotImplementedError  # pragma: no cover
