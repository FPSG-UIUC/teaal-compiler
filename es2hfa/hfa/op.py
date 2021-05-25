"""
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
    The HFA subtraction operator
    """

    def gen(self) -> str:
        """
        Generate the HFA code for the OSub operator
        """
        return "-"
