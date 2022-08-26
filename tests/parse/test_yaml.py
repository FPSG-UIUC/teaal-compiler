from teaal.parse.yaml import YamlParser


def test_parse_file():
    out = YamlParser.parse_file("tests/integration/example9.yaml")
    assert out["einsum"]["expressions"][0] == "T1[i, j] = sum(K, L).(A[i, j, l] * B[k, l])"


def test_parse_file_mapping():
    out = YamlParser.parse_file("tests/integration/example9.yaml")
    assert out["mapping"]["rank-order"]["A"] == ["J", "L", "I"]


def test_parse_str():
    yaml = """
    einsum:
        declaration:
            A: [I, J, L]
            B: [K, L]
            C: [I, J]
            D: [I]
            T1: [I, J]
        expressions:
            - T1[i, j] = sum(K, L).(A[i, j, l] * B[k, l])
            - D[i] = sum(J).(C[i, j] + T1[i, j])
    mapping:
        rank-order:
            A: [J, L, I]
            B: [L, K]
        partitions:
            D:
                J: [uniformShape(6), uniformShape(4)]
        loop-order:
            T1: [J, I, L, K]
    """
    out = YamlParser.parse_str(yaml)
    assert out["einsum"]["expressions"][0] == "T1[i, j] = sum(K, L).(A[i, j, l] * B[k, l])"
