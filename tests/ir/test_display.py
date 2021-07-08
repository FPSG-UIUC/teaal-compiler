import pytest

from es2hfa.ir.display import Display
from es2hfa.parse.display import DisplayParser
from tests.utils.parse_tree import make_uniform_shape


def create_yaml(space, time):
    space_trees = []
    for string in space:
        space_trees.append(DisplayParser.parse(string))

    time_trees = []
    for string in time:
        time_trees.append(DisplayParser.parse(string))

    return {"space": space_trees, "time": time_trees}


def test_bad_space():
    yaml = {"space": "a", "time": []}

    with pytest.raises(TypeError) as excinfo:
        Display(yaml, [], {}, "Z")
    assert str(
        excinfo.value) == "Display space argument must be a list, given a on output Z"


def test_bad_time():
    yaml = {"space": [], "time": "b"}

    with pytest.raises(TypeError) as excinfo:
        Display(yaml, [], {}, "Z")
    assert str(
        excinfo.value) == "Display time argument must be a list, given b on output Z"


def test_bad_spacetime():
    yaml = create_yaml(["N"], ["K", "N"])

    with pytest.raises(ValueError) as excinfo:
        Display(yaml, ["M", "N", "K"], {}, "Z")
    assert str(excinfo.value) == "Incorrect schedule for display on output Z"


def test_get_base():
    yaml = create_yaml(["N", "M"], ["K"])
    parts = {"M": make_uniform_shape([6, 3]), "N": make_uniform_shape([5])}
    display = Display(yaml, ["M", "N", "K"], parts, "Z")

    assert display.get_base("M0") == "M1"
    assert display.get_base("M1") == "M2"
    assert display.get_base("M2") is None
    assert display.get_base("K") is None


def test_get_space():
    yaml = create_yaml(["N", "M"], ["K"])
    display = Display(yaml, ["M", "N", "K"], {}, "Z")
    assert display.get_space() == ["M", "N"]


def test_get_style():
    yaml = create_yaml(["M"], ["K.pos", "N.coord"])
    display = Display(yaml, ["M", "N", "K"], {}, "Z")
    assert display.get_style("M") == "pos"
    assert display.get_style("K") == "pos"
    assert display.get_style("N") == "coord"


def test_get_time():
    yaml = create_yaml(["M"], ["K", "N"])
    display = Display(yaml, ["M", "N", "K"], {}, "Z")
    assert display.get_time() == ["N", "K"]


def test_eq():
    yaml = create_yaml(["M"], ["K", "N"])
    display1 = Display(yaml, ["M", "N", "K"], {}, "Z")
    display2 = Display(yaml, ["M", "N", "K"], {}, "Z")

    assert display1 == display2


def test_neq_space():
    yaml1 = create_yaml(["M"], ["K", "N"])
    display1 = Display(yaml1, ["M", "N", "K"], {}, "Z")

    yaml2 = create_yaml([], ["K", "M", "N"])
    display2 = Display(yaml2, ["M", "N", "K"], {}, "Z")

    assert display1 != display2


def test_neq_style():
    yaml1 = create_yaml(["M.pos"], ["K.pos", "N.pos"])
    display1 = Display(yaml1, ["M", "N", "K"], {}, "Z")

    yaml2 = create_yaml(["M.coord"], ["K.coord", "N.coord"])
    display2 = Display(yaml2, ["M", "N", "K"], {}, "Z")

    assert display1 != display2


def test_neq_time():
    yaml1 = create_yaml(["M"], ["K", "N"])
    display1 = Display(yaml1, ["M", "N", "K"], {}, "Z")

    yaml2 = create_yaml(["M"], ["K"])
    display2 = Display(yaml2, ["M", "K"], {}, "Z")

    assert display1 != display2


def test_neq_obj():
    yaml = create_yaml(["M"], ["K", "N"])
    display = Display(yaml, ["M", "N", "K"], {}, "Z")
    obj = "foo"

    assert display != obj
