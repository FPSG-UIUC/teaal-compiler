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

HFA AST and code generation for HFA expressions
"""

from teaal.hfa.base import Argument, Expression


class AJust(Argument):
    """
    An unparameterized argument to an HFA function
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self) -> str:
        """
        Generate the HFA code for an AJust
        """
        return self.expr.gen()


class AParam(Argument):
    """
    A parameterized argument to an HFA function
    """

    def __init__(self, name: str, expr: Expression) -> None:
        self.name = name
        self.expr = expr

    def gen(self) -> str:
        """
        Generate the HFA code for an AParam
        """
        return self.name + "=" + self.expr.gen()
