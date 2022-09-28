import pytest
from sympy import symbols

from teaal.ir.partitioning import Partitioning
from teaal.ir.spacetime import SpaceTime
from teaal.parse.mapping import Mapping
from teaal.parse.spacetime import SpaceTimeParser
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


def create_eqn_exprs():
    k, m, n = symbols("k m n")
    return {k: k, m: m, n: n}


def test_bad_space():
    yaml = {"space": "a", "time": []}
    eqn_exprs = create_eqn_exprs()

    with pytest.raises(TypeError) as excinfo:
        SpaceTime(yaml, Partitioning({}, [], eqn_exprs), "Z")
    assert str(
        excinfo.value) == "SpaceTime space argument must be a list, given a on output Z"


def test_bad_time():
    yaml = {"space": [], "time": "b"}
    eqn_exprs = create_eqn_exprs()

    with pytest.raises(TypeError) as excinfo:
        SpaceTime(yaml, Partitioning({}, [], eqn_exprs), "Z")
    assert str(
        excinfo.value) == "SpaceTime time argument must be a list, given b on output Z"


def test_bad_opt():
    yaml = {"space": [], "time": [], "opt": "foo"}
    eqn_exprs = create_eqn_exprs()

    with pytest.raises(ValueError) as excinfo:
        SpaceTime(yaml, Partitioning({}, [], eqn_exprs), "Z")
    assert str(excinfo.value) == "Unknown spacetime optimization foo on output Z"


def test_emit_pos_coord():
    yaml = create_yaml([], ["M.coord"])
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning({}, ["M"], eqn_exprs), "Z")
    assert not spacetime.emit_pos("M")


def test_emit_pos_no_opt():
    yaml = create_yaml([], ["M.pos"])
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning({}, ["M"], eqn_exprs), "Z")
    assert spacetime.emit_pos("M")


def test_emit_pos_slip_time():
    yaml = create_yaml([], ["M.pos"], "slip")
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning({}, ["M"], eqn_exprs), "Z")
    assert not spacetime.emit_pos("M")


def test_emit_pos_slip_space():
    yaml = create_yaml(["M.pos"], [], "slip")
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning({}, ["M"], eqn_exprs), "Z")
    assert spacetime.emit_pos("M")


def test_get_offset():
    yaml = create_yaml(["N", "M"], ["K"])
    eqn_exprs = create_eqn_exprs()

    part_yaml = """
    mapping:
        partitioning:
            Z:
                M: [uniform_shape(6), uniform_shape(3)]
                N: [uniform_shape(5)]
    """
    parts = Mapping.from_str(part_yaml).get_partitioning()["Z"]
    spacetime = SpaceTime(
        yaml, Partitioning(parts, ["M", "N", "K"], eqn_exprs), "Z")

    assert spacetime.get_offset("M0") == "M1"
    assert spacetime.get_offset("M1") == "M2"
    assert spacetime.get_offset("M2") is None
    assert spacetime.get_offset("K") is None


def test_get_slip_false():
    yaml = create_yaml(["N", "M"], ["K"])
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")
    assert not spacetime.get_slip()


def test_get_slip_true():
    yaml = create_yaml(["N", "M"], ["K"], "slip")
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")
    assert spacetime.get_slip()


def test_get_space():
    yaml = create_yaml(["N", "M"], ["K"])
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")
    assert spacetime.get_space() == ["N", "M"]


def test_get_style():
    yaml = create_yaml(["M"], ["K.pos", "N.coord"])
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")
    assert spacetime.get_style("M") == "pos"
    assert spacetime.get_style("K") == "pos"
    assert spacetime.get_style("N") == "coord"


def test_get_time():
    yaml = create_yaml(["M"], ["K", "N"])
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")
    assert spacetime.get_time() == ["K", "N"]


def test_eq():
    yaml = create_yaml(["M"], ["K", "N"])
    eqn_exprs = create_eqn_exprs()

    spacetime1 = SpaceTime(yaml, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")
    spacetime2 = SpaceTime(yaml, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")

    assert spacetime1 == spacetime2


def test_neq_space():
    eqn_exprs = create_eqn_exprs()

    yaml1 = create_yaml(["M"], ["K", "N"])
    spacetime1 = SpaceTime(yaml1, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")

    yaml2 = create_yaml([], ["K", "M", "N"])
    spacetime2 = SpaceTime(yaml2, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")

    assert spacetime1 != spacetime2


def test_neq_style():
    eqn_exprs = create_eqn_exprs()

    yaml1 = create_yaml(["M.pos"], ["K.pos", "N.pos"])
    spacetime1 = SpaceTime(yaml1, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")

    yaml2 = create_yaml(["M.coord"], ["K.coord", "N.coord"])
    spacetime2 = SpaceTime(yaml2, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")

    assert spacetime1 != spacetime2


def test_neq_time():
    eqn_exprs = create_eqn_exprs()

    yaml1 = create_yaml(["M"], ["K", "N"])
    spacetime1 = SpaceTime(yaml1, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")

    yaml2 = create_yaml(["M"], ["K"])
    spacetime2 = SpaceTime(yaml2, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")

    assert spacetime1 != spacetime2


def test_neq_obj():
    yaml = create_yaml(["M"], ["K", "N"])
    eqn_exprs = create_eqn_exprs()

    spacetime = SpaceTime(yaml, Partitioning(
        {}, ["M", "N", "K"], eqn_exprs), "Z")
    obj = "foo"

    assert spacetime != obj
