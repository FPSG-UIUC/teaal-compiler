import pytest

from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.spacetime import SpaceTime
from es2hfa.parse.spacetime import SpaceTimeParser
from tests.utils.parse_tree import make_uniform_shape


def create_yaml(space, time, opt=None):
    space_trees = []
    for string in space:
        space_trees.append(SpaceTimeParser.parse(string))

    time_trees = []
    for string in time:
        time_trees.append(SpaceTimeParser.parse(string))

    spacetime = {"space": space_trees, "time": time_trees}
    if opt is not None:
        spacetime["opt"] = opt

    return spacetime


def test_bad_space():
    yaml = {"space": "a", "time": []}

    with pytest.raises(TypeError) as excinfo:
        SpaceTime(yaml, [], Partitioning({}), "Z")
    assert str(
        excinfo.value) == "SpaceTime space argument must be a list, given a on output Z"


def test_bad_time():
    yaml = {"space": [], "time": "b"}

    with pytest.raises(TypeError) as excinfo:
        SpaceTime(yaml, [], Partitioning({}), "Z")
    assert str(
        excinfo.value) == "SpaceTime time argument must be a list, given b on output Z"


def test_bad_spacetime():
    yaml = create_yaml(["N"], ["K", "N"])

    with pytest.raises(ValueError) as excinfo:
        SpaceTime(yaml, ["M", "N", "K"], Partitioning({}), "Z")
    assert str(excinfo.value) == "Incorrect schedule for spacetime on output Z"


def test_bad_opt():
    yaml = {"space": [], "time": [], "opt": "foo"}
    with pytest.raises(ValueError) as excinfo:
        SpaceTime(yaml, [], Partitioning({}), "Z")
    assert str(excinfo.value) == "Unknown spacetime optimization foo on output Z"


def test_emit_pos_coord():
    yaml = create_yaml([], ["M.coord"])
    spacetime = SpaceTime(yaml, ["M"], Partitioning({}), "Z")
    assert not spacetime.emit_pos("M")


def test_emit_pos_no_opt():
    yaml = create_yaml([], ["M.pos"])
    spacetime = SpaceTime(yaml, ["M"], Partitioning({}), "Z")
    assert spacetime.emit_pos("M")


def test_emit_pos_slip_time():
    yaml = create_yaml([], ["M.pos"], "slip")
    spacetime = SpaceTime(yaml, ["M"], Partitioning({}), "Z")
    assert not spacetime.emit_pos("M")


def test_emit_pos_slip_space():
    yaml = create_yaml(["M.pos"], [], "slip")
    spacetime = SpaceTime(yaml, ["M"], Partitioning({}), "Z")
    assert spacetime.emit_pos("M")


def test_get_offset():
    yaml = create_yaml(["N", "M"], ["K"])
    parts = {"M": make_uniform_shape([6, 3]), "N": make_uniform_shape([5])}
    spacetime = SpaceTime(yaml, ["M", "N", "K"], Partitioning(parts), "Z")

    assert spacetime.get_offset("M0") == "M1"
    assert spacetime.get_offset("M1") == "M2"
    assert spacetime.get_offset("M2") is None
    assert spacetime.get_offset("K") is None


def test_get_slip_false():
    yaml = create_yaml(["N", "M"], ["K"])
    spacetime = SpaceTime(yaml, ["M", "N", "K"], Partitioning({}), "Z")
    assert not spacetime.get_slip()


def test_get_slip_true():
    yaml = create_yaml(["N", "M"], ["K"], "slip")
    spacetime = SpaceTime(yaml, ["M", "N", "K"], Partitioning({}), "Z")
    assert spacetime.get_slip()


def test_get_space():
    yaml = create_yaml(["N", "M"], ["K"])
    spacetime = SpaceTime(yaml, ["M", "N", "K"], Partitioning({}), "Z")
    assert spacetime.get_space() == ["N", "M"]


def test_get_style():
    yaml = create_yaml(["M"], ["K.pos", "N.coord"])
    spacetime = SpaceTime(yaml, ["M", "N", "K"], Partitioning({}), "Z")
    assert spacetime.get_style("M") == "pos"
    assert spacetime.get_style("K") == "pos"
    assert spacetime.get_style("N") == "coord"


def test_get_time():
    yaml = create_yaml(["M"], ["K", "N"])
    spacetime = SpaceTime(yaml, ["M", "N", "K"], Partitioning({}), "Z")
    assert spacetime.get_time() == ["K", "N"]


def test_eq():
    yaml = create_yaml(["M"], ["K", "N"])
    spacetime1 = SpaceTime(yaml, ["M", "N", "K"], Partitioning({}), "Z")
    spacetime2 = SpaceTime(yaml, ["M", "N", "K"], Partitioning({}), "Z")

    assert spacetime1 == spacetime2


def test_neq_space():
    yaml1 = create_yaml(["M"], ["K", "N"])
    spacetime1 = SpaceTime(yaml1, ["M", "N", "K"], Partitioning({}), "Z")

    yaml2 = create_yaml([], ["K", "M", "N"])
    spacetime2 = SpaceTime(yaml2, ["M", "N", "K"], Partitioning({}), "Z")

    assert spacetime1 != spacetime2


def test_neq_style():
    yaml1 = create_yaml(["M.pos"], ["K.pos", "N.pos"])
    spacetime1 = SpaceTime(yaml1, ["M", "N", "K"], Partitioning({}), "Z")

    yaml2 = create_yaml(["M.coord"], ["K.coord", "N.coord"])
    spacetime2 = SpaceTime(yaml2, ["M", "N", "K"], Partitioning({}), "Z")

    assert spacetime1 != spacetime2


def test_neq_time():
    yaml1 = create_yaml(["M"], ["K", "N"])
    spacetime1 = SpaceTime(yaml1, ["M", "N", "K"], Partitioning({}), "Z")

    yaml2 = create_yaml(["M"], ["K"])
    spacetime2 = SpaceTime(yaml2, ["M", "K"], Partitioning({}), "Z")

    assert spacetime1 != spacetime2


def test_neq_obj():
    yaml = create_yaml(["M"], ["K", "N"])
    spacetime = SpaceTime(yaml, ["M", "N", "K"], Partitioning({}), "Z")
    obj = "foo"

    assert spacetime != obj
