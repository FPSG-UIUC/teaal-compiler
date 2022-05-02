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

from typing import Dict, Sequence

from es2hfa.hfa.base import Argument, Expression, Operator


class EAccess(Expression):
    """
    An access into a list or dictionary
    """

    def __init__(self, obj: Expression, ind: Expression) -> None:
        self.obj = obj
        self.ind = ind

    def gen(self) -> str:
        """
        Generate the HFA code for an EAccess
        """
        return self.obj.gen() + "[" + self.ind.gen() + "]"


class EBinOp(Expression):
    """
    An HFA binary operation
    """

    def __init__(
            self,
            expr1: Expression,
            op: Operator,
            expr2: Expression) -> None:
        self.expr1 = expr1
        self.op = op
        self.expr2 = expr2

    def gen(self) -> str:
        """
        Generate the HFA code for an EBinOp
        """
        return self.expr1.gen() + " " + self.op.gen() + " " + self.expr2.gen()


class EBool(Expression):
    """
    An HFA boolean variable
    """

    def __init__(self, bool_: bool) -> None:
        self.bool = bool_

    def gen(self) -> str:
        """
        Generate the HFA code for an EBool
        """
        return str(self.bool)


class EComp(Expression):
    """
    An HFA list comprehension
    """

    def __init__(self, elem: Expression, var: str, iter_: Expression) -> None:
        self.elem = elem
        self.var = var
        self.iter = iter_

    def gen(self) -> str:
        """
        Generate the HFA code for an EComp
        """
        return "[" + self.elem.gen() + " for " + self.var + \
            " in " + self.iter.gen() + "]"


class EDict(Expression):
    """
    An HFA dictionary
    """

    def __init__(self, dict_: Dict[Expression, Expression]):
        self.dict = dict_

    def gen(self) -> str:
        """
        Generate the HFA code for an EDict
        """
        items = []
        for key, val in self.dict.items():
            items.append(key.gen() + ": " + val.gen())
        return "{" + ", ".join(items) + "}"


class EField(Expression):
    """
    An HFA object field access
    """

    def __init__(self, obj: str, field: str):
        self.obj = obj
        self.field = field

    def gen(self) -> str:
        """
        Generate the HFA code for an EField
        """
        return self.obj + "." + self.field


class EFloat(Expression):
    """
    An HFA float
    """

    def __init__(self, float_: float) -> None:
        self.float = float_

    def gen(self) -> str:
        """
        Generate HFA code for an EFloat
        """
        if self.float == float("inf"):
            return "float(\"inf\")"
        elif self.float == -float("inf"):
            return "-float(\"inf\")"
        else:
            return str(self.float)


class EFunc(Expression):
    """
    An HFA function call
    """

    def __init__(self, name: str, args: Sequence[Argument]) -> None:
        self.name = name
        self.args = args

    def gen(self) -> str:
        """
        Generate the HFA code for an EFunc
        """
        return self.name + \
            "(" + ", ".join([a.gen() for a in self.args]) + ")"


class EInt(Expression):
    """
    An HFA integer
    """

    def __init__(self, int_: int) -> None:
        self.int = int_

    def gen(self) -> str:
        """
        Generate HFA code for an EInt
        """
        return str(self.int)


class ELambda(Expression):
    """
    An HFA lambda
    """

    def __init__(self, args: Sequence[str], body: Expression) -> None:
        self.args = args
        self.body = body

    def gen(self) -> str:
        """
        Generate HFA code for an ELambda
        """
        return "lambda " + ", ".join(self.args) + ": " + self.body.gen()


class EList(Expression):
    """
    An HFA list
    """

    def __init__(self, list_: Sequence[Expression]) -> None:
        self.list = list_

    def gen(self) -> str:
        """
        Generate the HFA code for an EList
        """
        return "[" + ", ".join([e.gen() for e in self.list]) + "]"


class EMethod(Expression):
    """
    An HFA method call
    """

    def __init__(self, obj: Expression, name: str,
                 args: Sequence[Argument]) -> None:
        self.obj = obj
        self.name = name
        self.args = args

    def gen(self) -> str:
        """
        Generate the HFA code for an EMethod
        """
        return self.obj.gen() + "." + self.name + \
            "(" + ", ".join([a.gen() for a in self.args]) + ")"


class EParens(Expression):
    """
    An HFA expression surrounded by parentheses
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self) -> str:
        """
        Generate the HFA code for an EParens
        """
        return "(" + self.expr.gen() + ")"


class EString(Expression):
    """
    A string in HFA
    """

    def __init__(self, string: str) -> None:
        self.string = string

    def gen(self) -> str:
        """
        Generate the HFA code for an EString
        """
        return "\"" + self.string + "\""


class ETuple(Expression):
    """
    A tuple in HFA
    """

    def __init__(self, elems: Sequence[Expression]) -> None:
        self.elems = elems

    def gen(self) -> str:
        """
        Generate the HFA code for this tuple
        """
        # A single element tuple in Python needs an extra trailing comma
        if len(self.elems) == 1:
            return "(" + self.elems[0].gen() + ",)"

        else:
            return "(" + ", ".join([elem.gen() for elem in self.elems]) + ")"


class EVar(Expression):
    """
    An HFA variable
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def gen(self) -> str:
        """
        Generate the HFA code for an EVar
        """
        return self.name
