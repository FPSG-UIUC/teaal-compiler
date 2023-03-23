import pytest

from teaal.ir.equation import Equation
from teaal.ir.tensor import Tensor
from teaal.parse.einsum import Einsum
from teaal.parse.equation import EquationParser

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


def test_bad_einsum():
    einsum = EquationParser.parse("Z[m, n] = A[k, m] * B[k, n] + C[m, n]")
    tensors = {"Z": Tensor("Z", ["M", "N"]), "A": Tensor("A", ["K", "M"]),
               "B": Tensor("B", ["K", "N"]), "C": Tensor("C", ["M", "N"])}

    with pytest.raises(ValueError) as excinfo:
        Equation(einsum, tensors)
    assert str(
        excinfo.value) == "Malformed einsum: ensure all terms iterate over all ranks"


def test_undeclared_tensor():
    einsum = EquationParser.parse("Z[m, n] = A[k, m] * B[k, n]")
    tensors = {"Z": Tensor("Z", ["M", "N"]), "A": Tensor("A", ["K", "M"])}

    with pytest.raises(ValueError) as excinfo:
        Equation(einsum, tensors)
    assert str(
        excinfo.value) == "Undeclared tensor: B"


def test_get_output():
    equation = create_equation(0)
    tensor = equation.get_output()
    corr_tensor = Tensor("T", ["M", "N"])
    corr_tensor.set_is_output(True)

    assert tensor == corr_tensor


def test_get_tensors():
    equation = create_equation(0)
    tensors = equation.get_tensors()

    corr_tensors = []
    corr_tensors.append(Tensor("T", ["M", "N"]))
    corr_tensors[0].set_is_output(True)
    corr_tensors.append(Tensor("A", ["K", "M"]))
    corr_tensors.append(Tensor("B", ["K", "N"]))

    assert tensors == corr_tensors


def test_get_trees():
    equation = create_equation(0)
    trees = equation.get_trees()

    corr_trees = []
    corr_trees.append(make_output("T", ["m", "n"]))
    corr_trees.append(make_tensor("A", ["k", "m"]))
    corr_trees.append(make_tensor("B", ["k", "n"]))

    assert trees == corr_trees


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
