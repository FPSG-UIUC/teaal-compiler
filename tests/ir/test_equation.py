import pytest

from teaal.ir.equation import Equation
from teaal.ir.tensor import Tensor
from teaal.parse.einsum import Einsum
from teaal.parse.equation import EquationParser

from tests.utils.parse_tree import *


def create_equation(einsum):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            T: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - """ + einsum
    einsum = Einsum.from_str(yaml)
    tensors = {"Z": Tensor("Z", ["M", "N"]), "A": Tensor("A", ["K", "M"]),
               "B": Tensor("B", ["K", "N"]), "C": Tensor("C", ["M", "N"]),
               "T": Tensor("T", ["M", "N"])}

    return Equation(einsum.get_expressions()[0], tensors)


def create_matmul():
    return create_equation("Z[m, n] = A[k, m] * B[k, n]")


def create_complex():
    return create_equation(
        "Z[m, n] = A[k, m] * C[m, n] * d + take(B[k, n], T[m, n], e, 2)")


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


def test_repeated_tensor():
    einsum = EquationParser.parse("Z[m, n] = A[m, n] * A[m, n]")
    tensors = {"Z": Tensor("Z", ["M", "N"]), "A": Tensor("A", ["M", "N"])}

    with pytest.raises(ValueError) as excinfo:
        Equation(einsum, tensors)
    assert str(
        excinfo.value) == "Repeated tensor: A"


def test_get_factor_order():
    equation = create_complex()
    assert equation.get_factor_order() == {
        "A": (
            0, 1), "C": (
            0, 2), "d": (
                0, 0), "B": (
                    1, 0), "T": (
                        1, 1), "e": (
                            1, 2)}


def test_get_in_update():
    equation = create_complex()
    assert equation.get_in_update() == [[True, True, True], [
        False, False, True]]


def test_get_output():
    equation = create_matmul()
    tensor = equation.get_output()
    corr_tensor = Tensor("Z", ["M", "N"])
    corr_tensor.set_is_output(True)

    assert tensor == corr_tensor


def test_get_tensor_bad():
    equation = create_matmul()

    with pytest.raises(ValueError) as excinfo:
        equation.get_tensor("D")
    assert str(excinfo.value) == "Unknown tensor D"


def test_get_tensor_unused():
    equation = create_matmul()

    with pytest.raises(ValueError) as excinfo:
        equation.get_tensor("C")
    assert str(excinfo.value) == "Tensor C not used in this Einsum"


def test_get_tensor():
    equation = create_matmul()

    corr_tensors = []
    corr_tensors.append(Tensor("Z", ["M", "N"]))
    corr_tensors[0].set_is_output(True)
    corr_tensors.append(Tensor("A", ["K", "M"]))
    corr_tensors.append(Tensor("B", ["K", "N"]))

    assert equation.get_tensor("Z") == corr_tensors[0]
    assert equation.get_tensor("A") == corr_tensors[1]
    assert equation.get_tensor("B") == corr_tensors[2]


def test_get_tensors():
    equation = create_matmul()
    tensors = equation.get_tensors()

    corr_tensors = []
    corr_tensors.append(Tensor("Z", ["M", "N"]))
    corr_tensors[0].set_is_output(True)
    corr_tensors.append(Tensor("A", ["K", "M"]))
    corr_tensors.append(Tensor("B", ["K", "N"]))

    assert tensors == corr_tensors


def test_get_term_tensors():
    equation = create_complex()
    assert equation.get_term_tensors() == [["A", "C"], ["B", "T"]]


def test_get_term_vars():
    equation = create_complex()
    assert equation.get_term_vars() == [["d"], ["e"]]


def test_get_trees():
    equation = create_matmul()
    trees = equation.get_trees()

    corr_trees = []
    corr_trees.append(make_output("Z", ["m", "n"]))
    corr_trees.append(make_tensor("A", ["k", "m"]))
    corr_trees.append(make_tensor("B", ["k", "n"]))

    assert trees == corr_trees


def test_eq_true():
    equation1 = create_matmul()
    equation2 = create_matmul()
    assert equation1 == equation2


def test_eq_false():
    equation1 = create_matmul()
    equation2 = create_complex()
    assert equation1 != equation2


def test_eq_obj():
    equation = create_matmul()
    assert equation != "foo"
