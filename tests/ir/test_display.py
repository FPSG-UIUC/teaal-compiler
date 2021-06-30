import pytest

from es2hfa.ir.display import Display
from tests.utils.parse_tree import make_uniform_shape


def create_yaml(space, time, style):
    return {"space": space, "time": time, "style": style}


def test_bad_space():
    yaml = create_yaml("a", [], "")

    with pytest.raises(TypeError) as excinfo:
        Display(yaml, [], {}, "Z")
    assert str(
        excinfo.value) == "Display space argument must be a list, given a on output Z"


def test_bad_time():
    yaml = create_yaml([], "b", "")

    with pytest.raises(TypeError) as excinfo:
        Display(yaml, [], {}, "Z")
    assert str(
        excinfo.value) == "Display time argument must be a list, given b on output Z"


def test_bad_spacetime():
    yaml = create_yaml(["N"], ["K", "N"], "")

    with pytest.raises(ValueError) as excinfo:
        Display(yaml, ["M", "N", "K"], {}, "Z")
    assert str(excinfo.value) == "Incorrect schedule for display on output Z"


def test_bad_style_type():
    yaml = create_yaml([], [], [])

    with pytest.raises(TypeError) as excinfo:
        Display(yaml, [], {}, "Z")
    assert str(
        excinfo.value) == "Display style argument must be a string, given [] on output Z"


def test_bad_style_val():
    yaml = create_yaml([], [], "other")

    with pytest.raises(ValueError) as excinfo:
        Display(yaml, [], {}, "Z")
    assert str(excinfo.value) == "Unknown display style other on output Z"


def test_get_base():
    yaml = create_yaml(["N", "M"], ["K"], "shape")
    parts = {"M": make_uniform_shape([6, 3]), "N": make_uniform_shape([5])}
    display = Display(yaml, ["M", "N", "K"], parts, "Z")

    assert display.get_base("M0") == "M1"
    assert display.get_base("M1") == "M2"
    assert display.get_base("M2") is None
    assert display.get_base("K") is None


def test_get_space():
    yaml = create_yaml(["N", "M"], ["K"], "shape")
    display = Display(yaml, ["M", "N", "K"], {}, "Z")
    assert display.get_space() == ["M", "N"]


def test_get_style_shape():
    yaml = create_yaml(["M"], ["K", "N"], "shape")
    display = Display(yaml, ["M", "N", "K"], {}, "Z")
    assert display.get_style() == "shape"


def test_get_style_occupancy():
    yaml = create_yaml(["M"], ["K", "N"], "occupancy")
    display = Display(yaml, ["M", "N", "K"], {}, "Z")
    assert display.get_style() == "occupancy"


def test_get_time():
    yaml = create_yaml(["M"], ["K", "N"], "shape")
    display = Display(yaml, ["M", "N", "K"], {}, "Z")
    assert display.get_time() == ["N", "K"]


def test_eq():
    yaml = create_yaml(["M"], ["K", "N"], "shape")
    display1 = Display(yaml, ["M", "N", "K"], {}, "Z")
    display2 = Display(yaml, ["M", "N", "K"], {}, "Z")

    assert display1 == display2


def test_neq_space():
    yaml1 = create_yaml(["M"], ["K", "N"], "shape")
    display1 = Display(yaml1, ["M", "N", "K"], {}, "Z")

    yaml2 = create_yaml([], ["K", "M", "N"], "shape")
    display2 = Display(yaml2, ["M", "N", "K"], {}, "Z")

    assert display1 != display2


def test_neq_style():
    yaml1 = create_yaml(["M"], ["K", "N"], "shape")
    display1 = Display(yaml1, ["M", "N", "K"], {}, "Z")

    yaml2 = create_yaml(["M"], ["K", "N"], "occupancy")
    display2 = Display(yaml2, ["M", "N", "K"], {}, "Z")

    assert display1 != display2


def test_neq_time():
    yaml1 = create_yaml(["M"], ["K", "N"], "shape")
    display1 = Display(yaml1, ["M", "N", "K"], {}, "Z")

    yaml2 = create_yaml(["M"], ["K"], "shape")
    display2 = Display(yaml2, ["M", "K"], {}, "Z")

    assert display1 != display2


def test_neq_obj():
    yaml = create_yaml(["M"], ["K", "N"], "shape")
    display = Display(yaml, ["M", "N", "K"], {}, "Z")
    obj = "foo"

    assert display != obj
