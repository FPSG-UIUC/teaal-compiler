from es2hfa.ir.nodes import *


def test_node():
    assert Node() == Node()
    assert Node() != ""

    set_ = set()
    set_.add(Node())

    assert Node() in set_
    assert "" not in set_


def test_collecting_node():
    assert repr(CollectingNode("A", "K")) == "(CollectingNode, A, K)"

    assert CollectingNode("A", "K").get_tensor() == "A"
    assert CollectingNode("A", "K").get_rank() == "K"


def test_eager_input_node():
    assert repr(EagerInputNode("Q1", ["I", "J"])
                ) == "(EagerInputNode, Q1, ['I', 'J'])"

    assert EagerInputNode("Q1", ["I", "J"]).get_rank() == "Q1"
    assert EagerInputNode("Q1", ["I", "J"]).get_tensors() == ["I", "J"]


def test_fiber_node():
    assert repr(FiberNode("a_k")) == "(FiberNode, a_k)"

    assert FiberNode("a_k").get_fiber() == "a_k"


def test_from_fiber_node():
    assert repr(FromFiberNode("A", "K")) == "(FromFiberNode, A, K)"

    assert FromFiberNode("A", "K").get_tensor() == "A"
    assert FromFiberNode("A", "K").get_rank() == "K"


def test_get_root_node():
    assert repr(GetRootNode("A", ["K"])) == "(GetRootNode, A, ['K'])"

    assert GetRootNode("A", ["K"]).get_tensor() == "A"
    assert GetRootNode("A", ["K"]).get_ranks() == ["K"]


def test_interval_node():
    assert repr(IntervalNode("Q1")) == "(IntervalNode, Q1)"

    assert IntervalNode("Q1").get_rank() == "Q1"


def test_loop_node():
    assert repr(LoopNode("K1")) == "(LoopNode, K1)"

    assert LoopNode("K1").get_rank() == "K1"


def test_metrics_node():
    assert repr(MetricsNode("Start")) == "(MetricsNode, Start)"

    assert MetricsNode("Start").get_type() == "Start"


def test_other_node():
    assert repr(OtherNode("Foo")) == "(OtherNode, Foo)"

    assert OtherNode("Foo").get_type() == "Foo"


def test_part_node():
    assert repr(PartNode("A", ("K",))) == "(PartNode, A, ('K',))"

    assert PartNode("A", ("K",)).get_tensor() == "A"
    assert PartNode("A", ("K",)).get_ranks() == ("K",)


def test_rank_node():
    assert repr(RankNode("A", "K")) == "(RankNode, A, K)"

    assert RankNode("A", "K").get_tensor() == "A"
    assert RankNode("A", "K").get_rank() == "K"


def test_swizzle_root_node():
    assert repr(SwizzleNode("A", ["K"])) == "(SwizzleNode, A, ['K'])"

    assert SwizzleNode("A", ["K"]).get_tensor() == "A"
    assert SwizzleNode("A", ["K"]).get_ranks() == ["K"]


def test_tensor_node():
    assert repr(TensorNode("A")) == "(TensorNode, A)"

    assert TensorNode("A").get_tensor() == "A"
