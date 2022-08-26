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


from teaal.hfa.base import Operator


class OAdd(Operator):
    """
    The HFA addition operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OAdd operator
        """
        return "+"


class OAnd(Operator):
    """
    The HFA and operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OAnd operator
        """
        return "&"


class ODiv(Operator):
    """
    The HFA divide operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the ODiv operator
        """
        return "/"


class OEqEq(Operator):
    """
    The HFA equal-equal operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OEqEq operator
        """
        return "=="


class OFDiv(Operator):
    """
    The HFA floor divide operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OFDiv operator
        """
        return "//"


class OIn(Operator):
    """
    The HFA in operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OIn operator
        """
        return "in"


class OLt(Operator):
    """
    The HFA less-than less-than operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OLt operator
        """
        return "<"


class OLtLt(Operator):
    """
    The HFA less-than less-than operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OLtLt operator
        """
        return "<<"


class OMod(Operator):
    """
    The HFA modulo operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OMod operator
        """
        return "%"


class OMul(Operator):
    """
    The HFA multiplication operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OMul operator
        """
        return "*"


class OOr(Operator):
    """
    The HFA or operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OOr operator
        """
        return "|"


class OSub(Operator):
    """
    The HFA subtract operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OSub operator
        """
        return "-"
