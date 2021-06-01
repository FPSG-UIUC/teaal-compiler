from es2hfa.parse.yaml import YamlParser


def test_yaml_parse():
    out = YamlParser.parse("tests/integration/example9.yml")
    assert out["einsum"]["expressions"][0] == "T1[i, j] = sum(K, L).(A[i, j, l] * B[k, l])"


def test_yaml_parse_mapping():
    out = YamlParser.parse("tests/integration/example9.yml")
    assert out["mapping"]["rank-order"][0] == "A[J, L, I]"
