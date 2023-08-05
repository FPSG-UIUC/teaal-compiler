"""
MIT License

Copyright (c) 2021 University of Illinois

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Representations of all of the nodes in the FlowGraph
"""

import abc
from typing import Any, Iterable, List, Optional, Tuple

from teaal.ir.node import Node


class CollectingNode(Node):
    """
    A Node to turn on reuse distance collection for a particular rank of a
    tensor
    """

    def __init__(
            self,
            tensor: Optional[str],
            rank: str,
            type_: str,
            consumable: bool,
            is_read_trace: bool) -> None:
        """
        Construct a node for the collection of reuse metrics for a tensor's
        rank
        """
        self.tensor = tensor
        self.rank = rank
        self.type = type_
        self.consumable = consumable
        self.is_read_trace = is_read_trace

    def get_is_read_trace(self) -> bool:
        """
        Accessor for the is_read_trace property
        """
        return self.is_read_trace

    def get_consumable(self) -> bool:
        """
        Accessor for the consumable property
        """
        return self.consumable

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def get_tensor(self) -> Optional[str]:
        """
        Accessor for the tensor
        """
        return self.tensor

    def get_type(self) -> str:
        """
        Accessor for the type
        """
        return self.type

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a CollectingNode
        """
        return self.tensor, self.rank, self.type, self.consumable, self.is_read_trace


class ConsumeTraceNode(Node):
    """
    A node that consumes traces for a given component
    """

    def __init__(self, component: str, rank: str) -> None:
        """
        Construct a ConsumeTraceNode
        """
        self.component = component
        self.rank = rank

    def get_component(self) -> str:
        """
        Accessor for the component name
        """
        return self.component

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a ConsumeTraceNode
        """
        return self.component, self.rank


class CreateComponentNode(Node):
    """
    A node that creates an object for tracking metrics for a given component
    """

    def __init__(self, component: str, rank: str) -> None:
        """
        Construct a CreateComponentNode
        """
        self.component = component
        self.rank = rank

    def get_component(self) -> str:
        """
        Accessor for the component name
        """
        return self.component

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a CreateComponentNode
        """
        return self.component, self.rank


class EagerInputNode(Node):
    """
    A node that ensures that the inputs are eager
    """

    def __init__(self, rank: str, tensors: List[str]) -> None:
        """
        Construct a EagerInputNode
        """
        self.rank = rank
        self.tensors = tensors

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def get_tensors(self) -> List[str]:
        """
        Accessor for the tensor
        """
        return self.tensors

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a FromFiberNode
        """
        return self.rank, self.tensors


class EndLoopNode(Node):
    """
    A Node representing the end of a loop
    """

    def __init__(self, rank: str) -> None:
        """
        Construct a EndLoopNode
        """
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a EndLoopNode
        """
        return self.rank,


class FiberNode(Node):
    """
    A Node representing a fiber
    """

    def __init__(self, fiber: str) -> None:
        """
        Construct a FiberNode
        """
        self.fiber = fiber

    def get_fiber(self) -> str:
        """
        Accessor for the fiber
        """
        return self.fiber

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a FiberNode
        """
        return self.fiber,


class FromFiberNode(Node):
    """
    A Node representing a call to Tensor.fromFiber()
    """

    def __init__(self, tensor: str, rank: str) -> None:
        """
        Construct a FromFiberNode
        """
        self.tensor = tensor
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a FromFiberNode
        """
        return self.tensor, self.rank


class GetPayloadNode(Node):
    """
    A Node that represents a getPayload(Ref) call
    """

    def __init__(self, tensor: str, ranks: List[str]) -> None:
        """
        Construct a getPayload(Ref) node
        """
        self.tensor = tensor
        self.ranks = ranks

    def get_ranks(self) -> List[str]:
        """
        Accessor for the ranks
        """
        return self.ranks

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a GetPayloadNode
        """
        return self.tensor, self.ranks


class GetRootNode(Node):
    """
    A Node representing a getRoot call
    """

    def __init__(self, tensor: str, ranks: List[str]) -> None:
        """
        Construct a getRoot node
        """
        self.tensor = tensor
        self.ranks = ranks

    def get_ranks(self) -> List[str]:
        """
        Accessor for the ranks
        """
        return self.ranks

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a GetRootNode
        """
        return self.tensor, self.ranks


class IntervalNode(Node):
    """
    A Node representing the computation of the interval for projection of a
    fiber (or fibers)
    """

    def __init__(self, rank: str) -> None:
        """
        Construct an IntervalNode
        """
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of an IntervalNode
        """
        return self.rank,


class LoopNode(Node):
    """
    A Node representing a loop
    """

    def __init__(self, rank: str) -> None:
        """
        Construct a LoopNode
        """
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a LoopNode
        """
        return self.rank,


class MetricsFooterNode(Node):
    """
    A Node for collecting metrics before the start of the given loop

    TODO: Swallow everything necessary into this
    """

    def __init__(self, rank: str) -> None:
        """
        Construct a MetricsFooterNode
        """
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a MetricsFooterNode
        """
        return self.rank,


class MetricsHeaderNode(Node):
    """
    A Node for collecting metrics before the start of the given loop

    TODO: Swallow everything necessary into this
    """

    def __init__(self, rank: str) -> None:
        """
        Construct a MetricsHeaderNode
        """
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a MetricsHeaderNode
        """
        return self.rank,


class MetricsNode(Node):
    """
    A Node for metrics collection
    """

    def __init__(self, type_: str) -> None:
        """
        A node for metrics collection, type can be Start, End, or Dump
        """
        self.type = type_

    def get_type(self) -> str:
        """
        Accessor for the type
        """
        return self.type

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of the fields of an MetricsNode
        """
        return self.type,


class OtherNode(Node):
    """
    Another type of node
    """

    def __init__(self, type_: str) -> None:
        """
        Construct another type of node
        Should be used for required, one-off nodes
        """
        self.type = type_

    def get_type(self) -> str:
        """
        Accessor for the type
        """
        return self.type

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of the fields of an OtherNode
        """
        return self.type,


class PartNode(Node):
    """
    A Node representing a partitioning function
    """

    def __init__(self, tensor: str, ranks: Tuple[str, ...]) -> None:
        """
        Build a partitioning node for a given tensor and ranks
        """
        self.tensor = tensor
        self.ranks = ranks

    def get_ranks(self) -> Tuple[str, ...]:
        """
        Accessor for the ranks
        """
        return self.ranks

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a PartNode
        """
        return self.tensor, self.ranks


class RankNode(Node):
    """
    A Node representing a rank
    """

    def __init__(self, tensor: str, rank: str) -> None:
        """
        Construct a node for a rank name, tagged with its tensor
        """
        self.tensor = tensor
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a PartNode
        """
        return self.tensor, self.rank


class RegisterRanksNode(Node):
    """
    A node representing explicit rank registration
    """

    def __init__(self, ranks: List[str]) -> None:
        """
        Construct a node representing the explicit rank registration
        """
        self.ranks = ranks

    def get_ranks(self) -> List[str]:
        """
        Accessor for the ranks
        """
        return self.ranks

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a RegisterRanksNode
        """
        return self.ranks,


class SwizzleNode(Node):
    """
    A Node representing a swizzleRanks call
    """

    def __init__(self, tensor: str, ranks: List[str], type_: str) -> None:
        """
        Construct a swizzleRanks node
        """
        self.tensor = tensor
        self.ranks = ranks
        self.type = type_

    def get_ranks(self) -> List[str]:
        """
        Accessor for the ranks
        """
        return self.ranks

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def get_type(self) -> str:
        """
        Accessor for the type

        One of: "partitioning" or "loop-order"
        """
        return self.type

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a SwizzleNode
        """
        return self.tensor, self.ranks, self.type


class TensorNode(Node):
    """
    A Node representing a Tensor
    """

    def __init__(self, tensor: str) -> None:
        """
        Construct a tensor node
        """
        self.tensor = tensor

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a TensorNode
        """
        return self.tensor,


class TraceTreeNode(Node):
    """
    A Node representing the trace of a subtree
    """

    def __init__(self, tensor: str, rank: str, is_read_trace: bool):
        """
        Construct a TraceTreeNode
        """
        self.tensor = tensor
        self.rank = rank
        self.is_read_trace = is_read_trace

    def get_is_read_trace(self) -> bool:
        """
        Accessor for the is_read_trace property
        """
        return self.is_read_trace

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields fo a TraceTreeNode
        """
        return self.tensor, self.rank, self.is_read_trace
