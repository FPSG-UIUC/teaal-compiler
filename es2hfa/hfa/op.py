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

HFA AST and code generation for HFA operators
"""


from es2hfa.hfa.base import Operator


@Operator.register
class OAdd:
    """
    The HFA addition operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OAdd operator
        """
        return "+"


@Operator.register
class OAnd:
    """
    The HFA and operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OAnd operator
        """
        return "&"


@Operator.register
class OFDiv:
    """
    The HFA floor divide operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OFDiv operator
        """
        return "//"


@Operator.register
class OLtLt:
    """
    The HFA less-than less-than operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OLtLt operator
        """
        return "<<"


@Operator.register
class OMul:
    """
    The HFA multiplication operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OMul operator
        """
        return "*"


@Operator.register
class OOr:
    """
    The HFA or operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OOr operator
        """
        return "|"


@Operator.register
class OSub:
    """
    The HFA subtract operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OSub operator
        """
        return "-"
