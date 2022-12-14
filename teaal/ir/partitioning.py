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
from sympy import Basic, Symbol
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
                 eqn_exprs: Dict[Symbol, Basic]) -> None:
        """
        Create a new representation of the partitioning information
        """
        self.eqn_exprs = eqn_exprs

        self.__build_part_graph(partitioning, ranks)

        # Filter the partitioning information into the ranks that can
        # be partitioned statically vs dynamically
        self.dyn_parts = {}
        self.static_parts = {}

        for ranks_tree, parts in partitioning.items():
            # Continue if this rank is not actually partitioned
            if not parts:
                continue

            part_ranks = tuple(str(child) for child in ranks_tree.children)

            # Add the partitioning specification to the appropriate dictionary
            if Partitioning.__is_static(parts[0]):
                self.static_parts[part_ranks] = parts
            else:
                self.dyn_parts[part_ranks] = parts

        self.all_parts = {**self.static_parts, **self.dyn_parts}

        # Save the partitioning information for intermediate ranks
        # TODO: allow for more than one part_rank
        init_ranks = [part_ranks[0] for part_ranks in self.all_parts.keys()]
        for rank in init_ranks:
            for i, int_ in enumerate(self.get_intermediates(rank)):
                i = int(int_[len(rank):-1])
                self.dyn_parts[(int_,)] = self.all_parts[(rank,)][-i:]
                self.all_parts[(int_,)] = self.all_parts[(rank,)][-i:]

    def get_all_parts(self) -> Dict[Tuple[str, ...], List[Tree]]:
        """
        Get the partitioning information for all partitioned ranks
        """
        return self.all_parts

    def get_dyn_rank(self, rank: str) -> str:
        """
        Convert from a (potentially) static rank name to the corresponding
        dynamic rank name

        Used for the spacetime stamp of dynamically partitioned tensors
        """
        # TODO allow for flattened ranks
        ranks = []
        for key in self.dyn_parts.keys():
            ranks += list(key)
        if self.__tensor_to_part_rank(rank.upper(), ranks):
            return rank + "0"
        return rank

    def get_dyn_parts(self) -> Dict[Tuple[str, ...], List[Tree]]:
        """
        Get the partitioning information for all dynamically partitioned
        ranks
        """
        return self.dyn_parts

    def get_final_rank_id(self, rank: str) -> str:
        """
        Get the name of this rank in the final loop order
        """
        root_name = self.get_root_name(rank)
        # If this is an intermediate, translate it and find the final rank
        if rank[len(root_name):][-1:] == "I":
            # TODO allow for flattened ranks
            ranks = []
            for key in self.all_parts.keys():
                ranks += list(key)

            part_ranks = self.__tensor_to_part_rank(root_name, ranks)
            part_rank = Partitioning.__single_part_rank(root_name, part_ranks)

            return part_rank + rank[len(root_name):-1]

        # Otherwise, simply find the leaf rank
        node = self.nodes[rank]
        succ = list(self.graph.successors(node))
        while succ:
            node = max(succ, key=RankNode.get_priority)
            succ = list(self.graph.successors(node))

        return node.get_rank()

    def get_intermediates(self, rank: str) -> List[str]:
        """
        Get the names of all intermediate ranks (e.g., K2I)
        """
        # TODO allow for flattened ranks
        intermediates: List[str] = []
        node = None
        succ = list(self.graph.successors(self.nodes[rank]))
        while succ:
            if node:
                intermediates.append(node.get_rank())

            node = min(succ, key=RankNode.get_priority)
            succ = list(self.graph.successors(node))

        return intermediates

    def get_leader(self, part: Tree) -> str:
        """
        Return the leader tensor for this partitioning
        """
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

        part_num = self.nodes[rank].get_priority()

        # If this is not the top partition, return the offset
        offset = root_name + str(part_num + 1)
        if offset in self.nodes.keys():
            return offset

        return None

    def get_root_name(self, rank: str) -> str:
        """
        Get the root name for this partitioned rank (e.g., M1 -> M)
        """
        node = self.nodes[rank]
        pred = [n.get_rank() for n in self.graph.predecessors(node)]
        while pred:
            node = self.nodes[Partitioning.__best_match(rank, pred)]
            pred = [n.get_rank() for n in self.graph.predecessors(node)]

        return node.get_rank()

    def get_static_parts(self) -> Dict[Tuple[str, ...], List[Tree]]:
        """
        Get the partitioning information for all statically partitioned
        ranks
        """
        return self.static_parts

    def get_step(self, rank: str) -> Optional[str]:
        """
        Get the size of the step used to traverse over the given rank
        """
        # TODO: Raise error if the partition is dynamic
        # If this is an unpartitioned rank, return None
        root_name = self.get_root_name(rank)
        if rank == root_name:
            return None

        # Otherwise, get the partition number for this rank
        part_num = int(rank[len(root_name):])

        # If this is not the top partition, return the offset
        if part_num > 0:
            return root_name + str(part_num - 1)

        return None

    def get_tensor_spec(self, tensor_ranks: Iterable[str],
                        part_ranks: Iterable[str]) -> Dict[Tuple[str, ...], List[Tree]]:
        """
        Get the partitioning for a specific tensor's ranks
        """
        # TODO: after flattening is fixed, check if we still need the type hint
        partitioning: Dict[Tuple[str, ...], List[Tree]] = {}
        for ranks, part in self.all_parts.items():
            # TODO: allow for flattened ranks
            rank = ranks[0]

            if rank in part_ranks and self.__part_to_tensor_rank(
                    rank, tensor_ranks):
                # TODO: allow for flattened ranks
                partitioning[(rank,)] = part
        return partitioning

    def partition_names(self, rank: str, all_: bool) -> List[str]:
        """
        Get the list of names that this rank will be partitioned into
        """
        # TODO allow for flattened ranks
        if not all_:
            succ = list(self.graph.successors(self.nodes[rank]))

            if not succ:
                return [rank]

            return [
                node.get_rank() for node in sorted(
                    succ, key=RankNode.get_priority)]

        # Otherwise, do a depth-first traversal
        names = []
        curr = [self.nodes[rank]]
        while curr:
            node = curr.pop()
            succ = list(self.graph.successors(node))
            if not succ:
                names.append(node.get_rank())

            else:
                curr.extend(succ)

        return names

    def partition_rank(self, rank: str) -> Optional[str]:
        """
        Get the name of the corresponding partitioned rank, should one exist
        """
        # TODO allow for flattened ranks
        ranks = []
        for key in self.all_parts.keys():
            ranks += list(key)

        part_ranks = self.__tensor_to_part_rank(rank, ranks)
        if not part_ranks:
            return None

        return Partitioning.__single_part_rank(rank, part_ranks)

    def partition_tensor(
            self,
            tensor: Tensor,
            ranks: Iterable[str],
            all_: bool) -> List[str]:
        """
        Partition a tensor across all relevant ranks
        """
        all_parts = self.get_all_parts()
        tensor_ranks = tensor.get_ranks().copy()

        # TODO allow for flattened ranks
        all_ranks = []
        for key in self.all_parts.keys():
            all_ranks += list(key)

        for rank in tensor.get_ranks():
            # Check if there is anything to partition
            part_ranks = self.__tensor_to_part_rank(rank, all_ranks)
            if not part_ranks:
                continue

            # Check if we want to partition it
            part_rank = Partitioning.__single_part_rank(rank, part_ranks)
            if rank in ranks or part_rank in ranks:
                # Remove the old rank
                i = tensor_ranks.index(rank)
                tensor_ranks.pop(i)

                # Insert the new ranks
                new_ranks = self.partition_names(rank, all_)
                for new_rank in new_ranks:
                    tensor_ranks.insert(i, new_rank)

        return tensor_ranks

    def __add_or_update_priority(self, rank: str, priority: float) -> None:
        """
        Add the node if it does not exist, or update its priority
        """
        if rank not in self.nodes.keys():
            self.nodes[rank] = RankNode(rank, priority)
            return

        old_node = self.nodes[rank]
        if priority == old_node.get_priority():
            return

        preds = list(self.graph.predecessors(old_node))
        succs = list(self.graph.successors(old_node))

        self.graph.remove_node(old_node)

        new_node = RankNode(rank, priority)
        for pred in preds:
            self.graph.add_edge(pred, new_node)

        for succ in succs:
            self.graph.add_edge(new_node, succ)

        self.nodes[rank] = new_node

    @staticmethod
    def __best_match(rank: str, test_ranks: List[str]) -> str:
        """Find the best match for a rank given a list of options"""

        def prefix_match_len(rank0: str, rank1: str) -> int:
            """Find the number of characters that these two match for"""
            for i in range(min(len(rank0), len(rank1))):
                if rank0[i] != rank1[i]:
                    return i
            return min(len(rank0), len(rank1))

        best = None
        match_len = -float("inf")
        for test_rank in test_ranks:
            new_match_len = prefix_match_len(rank, test_rank)
            if new_match_len > match_len:
                best = test_rank
                match_len = new_match_len

        if best is None:
            raise ValueError(
                "Must be given at least one rank option")  # pragma: no cover

        return best

    def __build_part_graph(self,
                           partitioning: Dict[Tree, List[Tree]],
                           ranks: Iterable[str]) -> None:
        """
        Build the graph of how the partitioning information is related
        """

        self.graph = nx.DiGraph()
        self.nodes = {}

        # Add all of the starting ranks to the graph
        for rank in ranks:
            self.nodes[rank] = RankNode(rank, 0)
            self.graph.add_node(self.nodes[rank])

        # Some ranks used for partitioning may be created during partitioning
        # Add all needed ranks to ensure that they are available
        for ranks_tree in partitioning.keys():
            for rank_tree in ranks_tree.children:
                if str(rank_tree) not in self.nodes.keys():
                    self.nodes[rank] = RankNode(str(rank_tree), float("inf"))

        # Add the partitioning
        parted_by: Dict[str, List[str]] = {}
        roots: Dict[str, List[str]] = {}
        for ranks_tree, parts in partitioning.items():
            part_ranks = tuple(str(child) for child in ranks_tree.children)

            if Partitioning.__nway_after_dyn(parts):
                raise ValueError(
                    "N-way partitioning after dynamic partitioning on rank(s) " +
                    str(part_ranks))

            self.__check_flatten(part_ranks, parts, ranks)

            # If we are flattening, add a flattening node to combine them
            if len(part_ranks) > 1:
                flattened_rank = "".join(part_ranks)
                self.__add_or_update_priority(flattened_rank, 0)

                flatten_node = FlattenNode(part_ranks)
                self.graph.add_edge(flatten_node, self.nodes[flattened_rank])

                for part_rank in part_ranks:
                    self.graph.add_edge(self.nodes[part_rank], flatten_node)

                continue

            # Find the number of specifications used to partition each rank
            # Should be just 1
            part_rank = part_ranks[0]
            if part_rank not in roots.keys():
                roots[part_rank] = self.__part_to_tensor_rank(part_rank, ranks)

            for root in roots[part_rank]:
                if root not in parted_by:
                    parted_by[root] = []
                parted_by[root].append(part_rank)

            # Otherwise, divide the rank
            sources = [self.nodes[root_name] for root_name in roots[part_rank]]
            add_bottom_rank = False

            # Add edges
            for i, part in enumerate(parts):
                j = len(parts) - i
                if Partitioning.__is_static(part):
                    rank = part_rank + str(j)
                    self.__add_or_update_priority(rank, j)

                    for source in sources:
                        self.graph.add_edge(source, self.nodes[rank])

                    add_bottom_rank = True

                # Else dynamic partitioning
                else:
                    for k in range(len(sources)):
                        # If this is not the first partition, we need an
                        # explicit intermediate
                        if i > 0:
                            int_rank = roots[part_rank][k] + str(j) + "I"
                            self.__add_or_update_priority(int_rank, j)
                            self.graph.add_edge(
                                sources[k], self.nodes[int_rank])
                            sources[k] = self.nodes[int_rank]

                        rank = part_rank + str(j)
                        self.__add_or_update_priority(rank, j)
                        self.graph.add_edge(sources[k], self.nodes[rank])

                    add_bottom_rank = True

            if add_bottom_rank:
                for root_name, source in zip(roots[part_rank], sources):
                    rank = root_name + str(0)
                    self.__add_or_update_priority(rank, 0)
                    self.graph.add_edge(source, self.nodes[rank])

        # Ensure that each rank is only partitioned with one set of
        # partitioning information
        for rank, partitioners in parted_by.items():
            Partitioning.__single_part_rank(rank, partitioners)

    def __check_flatten(
            self, part_ranks: Tuple[str, ...], parts: List[Tree], all_ranks: Iterable[str]) -> None:
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
        for part in parts:
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

        if len(parts) > 1:
            raise ValueError(
                "flatten() combined with other operators on rank(s) " +
                str(part_ranks))

        if len(part_ranks) < 2:
            raise ValueError(
                "flatten() must combine at least two ranks; only " +
                str(part_ranks) +
                " specified")

        for rank in part_ranks:
            if len(self.__part_to_tensor_rank(rank, all_ranks)) > 1:
                raise ValueError(
                    "Cannot flatten rank " +
                    rank +
                    " because it is used in index math")

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
        return self.dyn_parts, self.static_parts, self.eqn_exprs

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

    def __part_to_tensor_rank(
            self,
            part_rank: str,
            tensor_ranks: Iterable[str]) -> List[str]:
        """
        Returns a tensor rank if one corresponds to the given partition rank
        """
        part_root = self.get_root_name(part_rank).lower()
        part_suffix = part_rank[len(part_root):]

        matches = []
        for tensor_rank in tensor_ranks:
            tensor_root = self.get_root_name(tensor_rank).lower()
            tensor_suffix = tensor_rank[len(tensor_root):]

            atoms = self.eqn_exprs[Symbol(tensor_root)].atoms(Symbol)
            if Symbol(part_root) in atoms and part_suffix == tensor_suffix:
                matches.append(tensor_rank)

        return matches

    @staticmethod
    def __single_part_rank(rank: str, part_ranks: List[str]) -> str:
        """
        Get a single part rank from a list
        Raises an error if there is not exactly one element in the list
        """
        if not part_ranks:
            raise ValueError("No partitioning for rank " + rank)

        elif len(part_ranks) > 1:
            raise ValueError(
                "Cannot partition " +
                rank +
                " with multiple specifications. Partitioning specified by " +
                ", ".join(part_ranks))

        return part_ranks[0]

    def __tensor_to_part_rank(
            self,
            tensor_rank: str,
            part_ranks: Iterable[str]) -> List[str]:
        """
        Return a partition rank if one corresponds to the given tensor rank
        """
        tensor_root = self.get_root_name(tensor_rank).lower()
        tensor_suffix = tensor_rank[len(tensor_root):]

        atoms = self.eqn_exprs[Symbol(tensor_root)].atoms(Symbol)
        eqn_ranks = {str(atom).upper() + tensor_suffix for atom in atoms}

        matches = []
        for eqn_rank in eqn_ranks:
            if eqn_rank in part_ranks:
                matches.append(eqn_rank)

        return matches
