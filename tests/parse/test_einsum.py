from teaal.parse.equation import EquationParser
from teaal.parse.einsum import Einsum


def test_declaration():
    einsum = Einsum.from_file("tests/integration/test_input.yaml")

    tensors = {
        "A": [
            "K", "M"], "B": [
            "K", "N"], "C": [
                "M", "N"], "T1": [
                    "M", "N"], "Z": [
                        "M", "N"]}
    assert einsum.get_declaration() == tensors


def test_eq():
    mapping = Einsum.from_file("tests/integration/test_input.yaml")
    assert mapping != "foo"


def test_expressions():
    mapping = Einsum.from_file("tests/integration/test_input.yaml")

    T1 = EquationParser.parse("T1[m, n] = sum(K).(A[k, m] * B[k, n])")
    Z = EquationParser.parse("Z[m, n] = T1[m, n] + C[m, n]")

    assert mapping.get_expressions() == [T1, Z]


def test_from():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            C: [M, N]
            T1: [M, N]
            Z: [M, N]
        expressions:
            - T1[m, n] = sum(K).(A[k, m] * B[k, n])
            - Z[m, n] = T1[m, n] + C[m, n]
    """
    from_file = Einsum.from_file("tests/integration/test_input.yaml")
    from_str = Einsum.from_str(yaml)
    assert from_file == from_str
