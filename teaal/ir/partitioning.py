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

Intermediate representation of the partitioning information
"""

from lark.tree import Tree
import networkx as nx  # type: ignore
from sympy import Basic, Symbol # type: ignore
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from teaal.ir.coord_math import CoordMath
from teaal.ir.part_nodes import *
from teaal.ir.tensor import Tensor
from teaal.parse.utils import ParseUtils


class Partitioning:
    """
    An abstract representation of the partitioning information
    """

    def __init__(self,
                 partitioning: Dict[Tree, List[Tree]],
                 ranks: Iterable[str],
                 coord_math: CoordMath) -> None:
        """
        Create a new representation of the partitioning information
        """
        self.orig_ranks = ranks
        self.coord_math = coord_math

        self.__build_part_graph(partitioning)

        # Filter the partitioning information into the ranks that can
        # be partitioned statically vs dynamically
        self.dyn_parts: Set[Tuple[str, ...]] = set()
        self.static_parts: Set[Tuple[str, ...]] = set()

        for node in nx.topological_sort(self.graph):
            if isinstance(node, FlattenNode):
                ranks = node.get_ranks()
                for rank in ranks:
                    if (self.get_root_name(rank),) in self.dyn_parts:
                        self.dyn_parts.add(ranks)
                        break

                if ranks not in self.dyn_parts:
                    self.static_parts.add(ranks)
                continue

            succs = [succ for succ in self.graph.successors(node)]
            if not succs:
                continue

            if isinstance(succs[0], FlattenNode):
                continue

            first = max(succs, key=lambda n: self.graph.nodes[n]["priority"])
            if Partitioning.__is_static(
                    self.graph.edges[(node, first)]["part"]):
                self.static_parts.add((node.get_rank(),))
            else:
                self.dyn_parts.add((node.get_rank(),))

        self.all_parts = self.static_parts.union(self.dyn_parts)

    def get_all_parts(self) -> Set[Tuple[str, ...]]:
        """
        Get the partitioning information for all partitioned ranks
        """
        return self.all_parts

    def get_available(self, rank: str) -> Set[str]:
        """
        Get the tensor ranks that may be available with this rank

        TODO: Cache this information
        """
        avail: Set[str] = set()
        avail.add(rank)

        frontier = [RankNode(rank)]
        while frontier:
            node = frontier.pop()
            preds = list(self.graph.predecessors(node))
            if not preds:
                continue

            assert len(preds) == 1
            parent = preds[0]

            if isinstance(parent, FlattenNode):
                # Ranks involved in flattening do not need to be translated
                avail.update(parent.get_ranks())
                frontier.extend(self.graph.predecessors(parent))
                continue

            min_child = min(
                self.graph.successors(parent),
                key=lambda n: self.graph.nodes[n]["priority"])
            if min_child == node:
                avail.add(parent.get_rank())
                frontier.append(parent)

        return avail

    def get_dyn_rank(self, rank: str) -> str:
        """
        Convert from a (potentially) static rank name to the corresponding
        dynamic rank name

        Used for the spacetime stamp of dynamically partitioned tensors
        """
        if self.is_flattened(rank):
            raise ValueError(
                "Should never be used for flattened ranks, used on rank " + rank)

        if RankNode(rank + "0") in self.graph.nodes:
            return rank + "0"

        return rank

    def get_dyn_parts(self) -> Set[Tuple[str, ...]]:
        """
        Get the partitioning information for all dynamically partitioned
        ranks
        """
        return self.dyn_parts

    def get_final_rank_id(self, init_ranks: Iterable[str], rank: str) -> str:
        """
        Get the name of this rank in the final loop order
        """
        # Find the leaf rank
        node: PartitioningNode = RankNode(rank)
        succ = list(self.graph.successors(node))
        comp = max
        while succ:
            if isinstance(succ[0], FlattenNode):
                assert len(succ) == 1
                node = succ[0]

                # If all flattened ranks appear in this tensor, the final rank
                # ID (location where this rank needs to be available) is the
                # top flattened rank
                # If all flattened ranks do not appear in the tensor, the final
                # rank ID is the bottom flattened rank
                for rank in node.get_ranks():
                    if self.get_root_name(rank) not in init_ranks:
                        comp = min
            else:
                node = comp(
                    succ, key=lambda n: self.graph.nodes[n]["priority"])

            succ = list(self.graph.successors(node))

        return node.get_rank()

    def get_intermediates(self, rank: str) -> List[str]:
        """
        Get the names of all intermediate split ranks (e.g., K2I)
        """
        intermediates: List[str] = []
        node = None
        succ = list(self.graph.successors(RankNode(rank)))
        while succ:
            if node and isinstance(succ[0], RankNode):
                intermediates.append(node.get_rank())

            if isinstance(succ[0], FlattenNode):
                break

            node = min(succ, key=lambda n: self.graph.nodes[n]["priority"])
            succ = list(self.graph.successors(node))

        return intermediates

    def get_leader(self, src_rank: str, dst_rank: str) -> str:
        """
        Return the leader tensor for this partitioning
        """
        part = self.graph.edges[(
            RankNode(src_rank), RankNode(dst_rank))]["part"]

        if part.data == "uniform_occupancy":
            return ParseUtils.find_str(part, "leader")

        raise ValueError("Style " + part.data + " has no leader")

    def get_offset(self, rank: str) -> Optional[str]:
        """
        Get the offset rank ID associated with a given rank, should one exist

        Used to convert from absolute to relavite coordinates
        """
        # If this is an unpartitioned rank, return None
        root_name = self.get_root_name(rank)
        if rank == root_name:
            return None

        part_num = self.graph.nodes[RankNode(rank)]["priority"]

        # If this is not the top partition, return the offset
        offset = root_name + str(part_num + 1)
        if self.graph.has_node(RankNode(offset)):
            return offset

        return None

    def get_root_name(self, rank: str) -> str:
        """
        Get the root name for this partitioned rank (e.g., M1 -> M)
        """
        node = RankNode(rank)

        if "root" in self.graph.nodes[node].keys():
            return self.graph.nodes[node]["root"]

        preds = [n.get_rank() for n in self.graph.predecessors(node)]
        while preds:
            assert len(preds) == 1
            node = RankNode(preds[0])

            preds = []
            for n in self.graph.predecessors(node):
                if isinstance(n, FlattenNode):
                    break

                preds.append(n.get_rank())

        nx.set_node_attributes(self.graph, {node: {"root": node.get_rank()}})
        return self.graph.nodes[node]["root"]

    def get_static_parts(self) -> Set[Tuple[str, ...]]:
        """
        Get the partitioning information for all statically partitioned
        ranks
        """
        return self.static_parts

    def get_step(self, rank: str) -> Optional[str]:
        """
        Get the size of the step used to traverse over the given rank
        """
        # If this is an unpartitioned rank, return None
        root_name = self.get_root_name(rank)
        if rank == root_name:
            return None

        # Check that this rank is statically partitioned
        if (root_name,) in self.dyn_parts:
            raise ValueError(
                "No static step for dynamically partitioned rank " + rank)

        # Otherwise, get the partition number for this rank
        part_num = int(rank[len(root_name):])

        # If this is not the top partition, return the offset
        if part_num > 0:
            return root_name + str(part_num - 1)

        return None

    def get_part_spec(self, part: Tuple[str, ...]) -> List[Tree]:
        """
        Get the partitioning specification for this partitioning for this tensor
        """
        spec: List[Tree] = []
        # If this is a splitting of a single rank into multiple
        if len(part) == 1:
            parent = RankNode(part[0])
            succs = [node for node in self.graph.successors(parent)]

            while succs:
                # Stop if we reach a FlattenNode (this should be covered
                # separately)
                if isinstance(succs[0], FlattenNode):
                    break

                # Note that the last and second to last partition will have the
                # same part, so we do not need to add it twice
                succs.sort(
                    key=lambda n: self.graph.nodes[n]["priority"],
                    reverse=True)
                for succ in succs[:-1]:
                    spec.append(self.graph.edges[(parent, succ)]["part"])

                parent = succs[-1]
                succs = [node for node in self.graph.successors(parent)]

        # Otherwise, this is a flattening of many into one
        else:
            flat_node = FlattenNode(part)
            rank_node = RankNode("".join(part))
            spec.append(self.graph.edges[(flat_node, rank_node)]["part"])

        return spec

    def get_valid_parts(self, ranks: List[str], parts: Iterable[Tuple[str, ...]],
                        allow_swizzle: bool) -> Iterable[Tuple[str, ...]]:
        """
        Get the valid partitionings for a given set of ranks
        """
        ranks = ranks.copy()
        used_parts: List[Tuple[str, ...]] = []
        new_parts = self.__used_parts(ranks, parts, allow_swizzle)

        while new_parts:
            used_parts.extend(new_parts)
            for part in new_parts:
                self.__update_ranks(part, ranks, False)

            new_parts = self.__used_parts(ranks, parts, allow_swizzle)

        return used_parts

    def is_flattened(self, rank: str) -> bool:
        """
        Return true if the rank is the result of a flattening
        """
        node = RankNode(rank)

        if "is_flattened" in self.graph.nodes[node].keys():
            return self.graph.nodes[node]["is_flattened"]

        preds = [n.get_rank() for n in self.graph.predecessors(node)]
        while preds:
            assert len(preds) == 1
            node = RankNode(preds[0])
            preds = []
            for n in self.graph.predecessors(node):
                if isinstance(n, FlattenNode):
                    self.graph.nodes[node]["is_flattened"] = True
                    return True
                elif isinstance(n, RankNode):
                    preds.append(n.get_rank())
                else:
                    raise ValueError(
                        "Unknown partitioning node type " + str(type(n.__name__)))  # pragma: no cover

        self.graph.nodes[node]["is_flattened"] = False
        return False

    def partition_names(self, ranks: Tuple[str, ...], all_: bool) -> List[str]:
        """
        Get the list of names that these ranks will be partitioned into
        """
        # Ensure there is at least 1 rank
        if not ranks:
            raise ValueError("At least one rank required")

        # Otherwise, traverse the partitioning graph
        frontier: List[Tuple[PartitioningNode, int]]
        priorities = {}
        if len(ranks) == 1:
            frontier = [(RankNode(ranks[0]), 0)]
            priorities[ranks[0]] = self.graph.nodes[RankNode(
                ranks[0])]["priority"]
        else:
            frontier = [(FlattenNode(ranks), 0)]

        names = []
        while frontier:
            node, depth = frontier.pop()
            is_leaf = True
            for succ in self.graph.successors(node):
                if isinstance(succ, FlattenNode):
                    continue

                if all_ or depth == 0:
                    frontier.append((succ, depth + 1))
                    is_leaf = False

            if is_leaf:
                names.append(node.get_rank())
                priorities[node.get_rank()] = self.graph.nodes[node]["priority"]

        names.sort(key=lambda n: priorities[n])
        return names

    def partition_rank(
            self, ranks: Tuple[str, ...]) -> Optional[Tuple[str, ...]]:
        """
        Get the name of the corresponding partitioned rank, should one exist
        """
        if len(ranks) > 1:
            if ranks in self.all_parts:
                return ranks
            return None

        rank = ranks[0]
        root = self.get_root_name(rank)

        if root in self.part_rank.keys():
            return (self.part_rank[root] + rank[len(root):],)

        return None

    def partition_ranks(
            self,
            tensor_ranks: List[str],
            allowed_parts: Iterable[Tuple[str, ...]],
            all_levels: bool,
            allow_swizzle: bool) -> List[str]:
        """
        Partition a tensor across all relevant partitionable ranks
        """
        tensor_ranks = tensor_ranks.copy()

        used_parts = self.__used_parts(
            tensor_ranks, allowed_parts, allow_swizzle)
        while used_parts:
            for part_ranks in used_parts:
                self.__update_ranks(part_ranks, tensor_ranks, all_levels)

            if not all_levels:
                break

            used_parts = self.__used_parts(
                tensor_ranks, allowed_parts, allow_swizzle)

        return tensor_ranks

    def split_rank_name(self, rank: str) -> Tuple[str, str]:
        """
        Split the rank name into the root and the suffix
        """
        root = self.get_root_name(rank)
        return root, rank[len(root):]

    def swizzle_for_flattening(self, tensor_ranks: List[str]) -> List[str]:
        """
        Swizzle the tensor ranks to allow for flattening
        """
        new_ranks = tensor_ranks.copy()
        used_parts = self.__used_parts(new_ranks, self.all_parts, True)
        for part in used_parts:
            if len(part) > 1:
                for rank in part:
                    del new_ranks[new_ranks.index(rank)]
                    new_ranks.append(rank)

        return new_ranks

    def unpack(self, rank: str) -> Tuple[str, ...]:
        """
        Unpack a flattened rank
        """
        if not self.is_flattened(rank):
            raise ValueError("Nothing to unpack for rank " + rank)

        node = RankNode(rank)
        preds = list(self.graph.predecessors(node))
        while len(preds) == 1:
            preds = list(self.graph.predecessors(preds[0]))

        return tuple(node.get_rank() for node in preds)

    def __add_or_update_priority(self, rank: str, priority: float) -> None:
        """
        Add the node if it does not exist, or update its priority
        """
        node = RankNode(rank)
        if node not in self.graph.nodes:
            self.graph.add_node(node, priority=priority)
        else:
            self.graph.nodes[node]["priority"] = priority

    def __build_part_graph(self, partitioning: Dict[Tree, List[Tree]]) -> None:
        """
        Build the graph of how the partitioning information is related
        """

        self.graph = nx.DiGraph()
        ranks = set(self.orig_ranks)

        # Add all of the starting ranks to the graph
        for rank in ranks:
            self.__add_or_update_priority(rank, 0)

        # Some ranks used for partitioning may be created during partitioning
        # Add all needed ranks to ensure that they are available
        for ranks_tree in partitioning.keys():
            for rank_tree in ranks_tree.children:
                rank = str(rank_tree)
                if not self.graph.has_node(RankNode(rank)):
                    self.__add_or_update_priority(rank, float("inf"))

        # Add the partitioning
        roots: Dict[str, List[str]] = {}
        edge: Tuple[PartitioningNode, PartitioningNode]
        all_parts = {tuple(str(child) for child in ranks_tree.children): parts
                     for ranks_tree, parts in partitioning.items()}

        self.part_rank = {}
        for part_ranks, parts in all_parts.items():
            # If there is nothing to partition, move on
            if not parts:
                continue

            if Partitioning.__nway_after_dyn(parts):
                raise ValueError(
                    "N-way partitioning after dynamic partitioning on rank(s) " +
                    str(part_ranks))

            self.__check_flatten(part_ranks, all_parts, ranks)

            # If we are flattening, add a flattening node to combine them
            if len(part_ranks) > 1:
                flattened_rank = "".join(part_ranks)
                self.__add_or_update_priority(flattened_rank, 0)

                flatten_node = FlattenNode(part_ranks)
                edge = (flatten_node, RankNode(flattened_rank))
                self.graph.add_edge(*edge, part=parts[0])

                for part_rank in part_ranks:
                    edge = (RankNode(part_rank), flatten_node)
                    self.graph.add_edge(*edge, part=parts[0])

                ranks.add(flattened_rank)

                continue

            # Otherwise, divide the rank
            source_name = part_ranks[0]
            source_node = RankNode(source_name)

            # If this is a follower rank, then replace the partitioning
            if parts[0].data == "follow":
                leader = ParseUtils.next_str(parts[0])
                parts = all_parts[(leader,)]

            else:
                leader = source_name
            self.part_rank[source_name] = leader

            # Add edges
            for i, part in enumerate(parts):
                j = len(parts) - i
                if Partitioning.__is_static(part):
                    if source_name not in self.orig_ranks:
                        raise ValueError(
                            "Shape-based partitioning found on rank " +
                            source_name +
                            " after flattening")

                    rank = source_name + str(j)
                    self.__add_or_update_priority(rank, j)

                    edge = (source_node, RankNode(rank))
                    self.graph.add_edge(*edge, part=part)

                    continue

                # Else dynamic partitioning

                # If this is not the first partition, we need an
                # explicit intermediate
                if i > 0:
                    int_rank = source_name + str(j) + "I"
                    self.__add_or_update_priority(int_rank, j)

                    edge = (source_node, RankNode(int_rank))
                    self.graph.add_edge(*edge, part=parts[i - 1])

                    source_node = RankNode(int_rank)

                rank = source_name + str(j)
                self.__add_or_update_priority(rank, j)

                edge = (source_node, RankNode(rank))
                self.graph.add_edge(*edge, part=part)

            # Add the bottom rank if needed
            if parts:
                rank = source_name + str(0)
                self.__add_or_update_priority(rank, 0)

                edge = (source_node, RankNode(rank))
                self.graph.add_edge(*edge, part=parts[-1])

    def __check_flatten(self, part_ranks: Tuple[str, ...], all_parts: Dict[Tuple[str, ...],
                        List[Tree]], all_ranks: Iterable[str]) -> None:
        """
        Check all conditions associated with flattening, and raise the
        appropriate errors

        Conditions:
        1. If a flatten() has been specified, it is the only operator specified
           with the given ranks
        2. Multiple ranks are being flattened together
        3. None of the flattened ranks are involved in index math
        """
        flatten = False
        ops = []
        for part in all_parts[part_ranks]:
            ops.append(part.data)
            if part.data == "flatten":
                flatten = True
                break

        if not flatten:
            if len(part_ranks) != 1:
                raise ValueError(
                    "Operations " +
                    str(ops) +
                    " can only be applied to one rank; not " +
                    str(part_ranks))

            return

        if len(all_parts[part_ranks]) > 1:
            raise ValueError(
                "flatten() combined with other operators on rank(s) " +
                str(part_ranks))

        if len(part_ranks) < 2:
            raise ValueError(
                "flatten() must combine at least two ranks; only " +
                str(part_ranks) +
                " specified")

        for rank in part_ranks:
            if len(self.coord_math.get_all_exprs(rank.lower())) > 1:
                raise ValueError(
                    "Cannot flatten rank " +
                    rank +
                    " because it is used in index math")

        # Necessary to ensure concordant traversal
        for rank in part_ranks:
            if (rank,) in all_parts and all_parts[(rank,)]:
                raise ValueError(
                    "Cannot flatten rank " +
                    rank +
                    " because it will also be independently partitioned")

            if rank not in self.orig_ranks:
                if rank in all_ranks:
                    raise ValueError(
                        "Cannot flatten rank " +
                        rank +
                        " because it is a flattened rank")

                if rank[:-1] not in self.orig_ranks or rank[-1] != "0":
                    raise ValueError(
                        "Cannot flatten rank " +
                        rank +
                        " because it will have multiple partitionings")

    def __eq__(self, other) -> bool:
        """
        The == operator for Partitionings
        """
        if isinstance(other, type(self)):
            return self.__key() == other.__key()

        return False

    @staticmethod
    def __is_static(part: Tree) -> bool:
        """
        Return true if this style of partitioning can be performed statically
        """
        return part.data == "uniform_shape" or part.data == "nway_shape"

    def __key(self) -> Iterable[Any]:
        """
        Get all relevant fields of the Partitioning
        """
        edges = {edge: self.graph.edges[edge]["part"]
                 for edge in self.graph.edges}
        return set(self.graph.nodes), edges

    @staticmethod
    def __nway_after_dyn(parts: List[Tree]) -> bool:
        """
        Check that this combination of partitionings does not have an nway
        partitioning after a dynamic partitioning
        """
        dyn = False
        for part in parts:
            if not Partitioning.__is_static(part):
                dyn = True
            elif part.data == "nway_shape" and dyn:
                return True

        return False

    def __update_ranks(
            self, part_ranks: Tuple[str, ...], tensor_ranks: List[str], all_: bool) -> None:
        """
        Update a list of ranks with a given partitioning
        Note: performs an in-place update
        """
        i = tensor_ranks.index(part_ranks[0])
        in_place = True
        for part_rank in part_ranks:
            if i >= len(tensor_ranks) or tensor_ranks[i] != part_rank:
                in_place = False
            tensor_ranks.remove(part_rank)

        if not in_place:
            i = len(tensor_ranks)

        new_ranks = self.partition_names(part_ranks, all_)
        for new_rank in new_ranks:
            tensor_ranks.insert(i, new_rank)

    def __used_parts(self, tensor_ranks: List[str], parts: Iterable[Tuple[str, ...]],
                     allow_swizzle: bool) -> Iterable[Tuple[str, ...]]:
        """
        Get the partitions used to partition the given ranks
        """
        used_parts = []
        for part_ranks in parts:
            # Check if this partitioning is used
            used = True
            next_ind = None
            new_part_ranks = []
            for part_rank in part_ranks:
                tensor_rank = part_rank
                if tensor_rank not in tensor_ranks:
                    used = False
                    break

                if next_ind is None:
                    next_ind = tensor_ranks.index(tensor_rank) + 1
                    new_part_ranks.append(tensor_rank)
                elif allow_swizzle or (next_ind < len(tensor_ranks) and tensor_ranks[next_ind] == tensor_rank):
                    next_ind += 1
                    new_part_ranks.append(tensor_rank)
                else:
                    used = False
                    break

            if used:
                used_parts.append(tuple(new_part_ranks))

        return used_parts
