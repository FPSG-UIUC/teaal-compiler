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

HiFiber AST and code generation for HiFiber operators
"""


from teaal.hifiber.base import Operator


class OAdd(Operator):
    """
    The HiFiber addition operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OAdd operator
        """
        return "+"


class OAnd(Operator):
    """
    The HiFiber and operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OAnd operator
        """
        return "&"


class ODiv(Operator):
    """
    The HiFiber divide operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the ODiv operator
        """
        return "/"


class OEqEq(Operator):
    """
    The HiFiber equal-equal operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OEqEq operator
        """
        return "=="


class OFDiv(Operator):
    """
    The HiFiber floor divide operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OFDiv operator
        """
        return "//"


class OIn(Operator):
    """
    The HiFiber in operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OIn operator
        """
        return "in"


class OLt(Operator):
    """
    The HiFiber less-than less-than operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OLt operator
        """
        return "<"


class OLtLt(Operator):
    """
    The HiFiber less-than less-than operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OLtLt operator
        """
        return "<<"


class OMod(Operator):
    """
    The HiFiber modulo operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OMod operator
        """
        return "%"


class OMul(Operator):
    """
    The HiFiber multiplication operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OMul operator
        """
        return "*"


class ONotIn(Operator):
    """
    The HiFiber not in operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the ONotIn operator
        """
        return "not in"


class OOr(Operator):
    """
    The HiFiber or operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OOr operator
        """
        return "|"


class OSub(Operator):
    """
    The HiFiber subtract operator
    """

    def gen(self) -> str:
        """
        Generate the HiFiber code for the OSub operator
        """
        return "-"
