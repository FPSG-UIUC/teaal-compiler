import pytest

from teaal.ir.equation import Equation
from teaal.ir.tensor import Tensor
from teaal.parse.einsum import Einsum

from tests.utils.parse_tree import *


def create_equation(i):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            T: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - T[m, n] = A[k, m] * B[k, n]
            - Z[m, n] = T[m, n] + C[m, n]
    """
    einsum = Einsum.from_str(yaml)
    tensors = {"Z": Tensor("Z", ["M", "N"]), "A": Tensor("A", ["K", "M"]),
               "B": Tensor("B", ["K", "N"]), "C": Tensor("C", ["M", "N"]),
               "T": Tensor("T", ["M", "N"])}

    return Equation(einsum.get_expressions()[i], tensors)


def test_get_output():
    equation = create_equation(0)
    tensor, tree = equation.get_output()

    corr_tensor = Tensor("T", ["M", "N"])
    corr_tensor.set_is_output(True)
    corr_tree = make_output("T", ["m", "n"])

    assert tensor == corr_tensor
    assert tree == corr_tree


def test_get_tensors():
    equation = create_equation(0)
    tensors = equation.get_tensors()

    corr_tensors = []
    corr_tensors.append(
        (Tensor(
            "A", [
                "K", "M"]), make_tensor(
            "A", [
                "k", "m"])))
    corr_tensors.append(
        (Tensor(
            "B", [
                "K", "N"]), make_tensor(
            "B", [
                "k", "n"])))

    assert tensors == corr_tensors


def test_eq_true():
    equation1 = create_equation(0)
    equation2 = create_equation(0)
    assert equation1 == equation2


def test_eq_false():
    equation1 = create_equation(0)
    equation2 = create_equation(1)
    assert equation1 != equation2


def test_eq_obj():
    equation = create_equation(0)
    assert equation != "foo"
