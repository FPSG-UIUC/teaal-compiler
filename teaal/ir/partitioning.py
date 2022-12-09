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

            if Partitioning.__nway_after_dyn(parts):
                raise ValueError(
                    "N-way partitioning after dynamic partitioning on rank " +
                    part_ranks[0])

            # Add the partitioning specification to the appropriate dictionary
            if Partitioning.__is_static(parts[0]):
                self.static_parts[part_ranks] = parts
            else:
                self.dyn_parts[part_ranks] = parts

        self.all_parts = {**self.static_parts, **self.dyn_parts}

        # All possible rank IDs to the final rank ID
        self.final_rank_id = {}

        # Unpartitioned ranks will not change
        for rank in ranks:
            self.final_rank_id[rank] = rank

        # Partitioned ranks may change
        for part_ranks in self.all_parts.keys():
            if len(part_ranks) > 1:
                raise ValueError("TODO: not yet ready")
            part_rank = part_ranks[0]

            part_names = self.partition_names(part_rank, True)
            # Add partitioned to itself, e.g., K0 -> K0
            for id_ in part_names:
                self.final_rank_id[id_] = id_

            for rank in self.__part_to_tensor_rank(part_rank, ranks):
                # Add unpartitioned to top rank, e.g., K -> K2 or W -> Q2
                self.final_rank_id[rank] = part_names[-1]

                # Add intermediate to final names, e.g., K1I -> K1 or W1I -> Q1
                for id_ in self.get_intermediates(rank):
                    self.final_rank_id[id_] = part_rank + id_[len(rank):-1]

                # Add the bottom rank to itself, e.g., W0 -> W0
                self.final_rank_id[rank + "0"] = rank + "0"

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
        return self.final_rank_id[rank]

    def get_intermediates(self, rank: str) -> List[str]:
        """
        Get the names of all intermediate ranks (e.g., K2I)
        """
        # TODO allow for flattened ranks
        ranks = []
        for key in self.all_parts.keys():
            ranks += list(key)

        part_ranks = self.__tensor_to_part_rank(rank, ranks)
        part_rank = Partitioning.__single_part_rank(rank, part_ranks)

        intermediates = []
        # TODO: allow for flattened ranks
        for i, part in enumerate(self.all_parts[(part_rank,)][1:]):
            num = len(self.all_parts[(part_rank,)]) - i - 1
            if not Partitioning.__is_static(part):
                intermediates.append(rank + str(num) + "I")

        return intermediates

    def get_leader(self, part: Tree) -> str:
        """
        Return the leader tensor for this partitioning
        """
        if part.data == "uniform_occupancy":
            return ParseUtils.find_str(part, "leader")

        raise ValueError("Style " + part.data + " has no leader")

    def get_root_name(self, rank: str) -> str:
        """
        Get the root name for this partitioned rank (e.g., M1 -> M)
        """
        node = self.nodes[rank]
        curr = node
        pred = [node]
        while pred:
            # TODO: Support flattening
            if len(pred) != 1:
                raise NotImplementedError

            curr = pred[0]
            pred = list(self.graph.predecessors(pred[0]))

        return curr.get_rank()


    def get_static_parts(self) -> Dict[Tuple[str, ...], List[Tree]]:
        """
        Get the partitioning information for all statically partitioned
        ranks
        """
        return self.static_parts

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
        ranks = []
        for key in self.all_parts.keys():
            ranks += list(key)

        part_ranks = self.__tensor_to_part_rank(rank, ranks)
        part_rank = Partitioning.__single_part_rank(rank, part_ranks)

        # TODO allow for flattened ranks
        parts = self.all_parts[(part_rank,)]
        # Return all final rank names
        if all_:
            parted_ = [part_rank + str(j) for j in range(len(parts) + 1)]
            # The bottom rank is named with the original rank name
            parted_[0] = rank + "0"
            return parted_

        part_root = self.get_root_name(part_rank)
        names = [part_root + str(len(parts))]

        root = self.get_root_name(rank)

        early_exit = False
        # Otherwise, add rank names until another dynamic partitioning
        for i, part in enumerate(parts[1:]):
            num = len(parts) - i - 1
            if Partitioning.__is_static(part):
                names.append(part_root + str(num))
                continue

            names.append(root + str(num) + "I")
            early_exit = True
            break

        if not early_exit:
            names.append(root + "0")
        names.reverse()

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

        # Add the partitioning
        for ranks_tree, parts in partitioning.items():
            part_ranks = tuple(str(child) for child in ranks_tree.children)

            if Partitioning.__nway_after_dyn(parts):
                raise ValueError(
                    "N-way partitioning after dynamic partitioning on rank(s) " +
                    str(part_ranks))


            # TODO: Support flattening
            roots = self.__part_to_tensor_rank(part_ranks[0], ranks)
            for root_name in roots:

                # Add edges
                # TODO: Support flattening
                sources = [self.nodes[root_name]]
                add_bottom_rank = False
                for i, part in enumerate(parts):
                    j = len(parts) - i
                    if Partitioning.__is_static(part):
                        if len(sources) != 1:
                            raise ValueError(part.data + " can only be performed on a single rank")

                        rank = root_name + str(j)
                        self.nodes[rank] = RankNode(rank, j)
                        self.graph.add_edge(sources[0], self.nodes[rank])

                        add_bottom_rank = True

                    # Else dynamic partitioning
                    # TODO: Support flattening
                    else:
                        if len(sources) != 1:
                            raise ValueError(part.data + " can only be performed on a single rank")

                        int_rank = root_name + str(j) + "I"
                        self.nodes[int_rank] = RankNode(int_rank, j)
                        self.graph.add_edge(sources[0], self.nodes[int_rank])
                        sources = [self.nodes[int_rank]]

                        rank = root_name + str(j)
                        self.nodes[rank] = RankNode(rank, j)
                        self.graph.add_edge(sources[0], self.nodes[rank])

                        add_bottom_rank = True

                if add_bottom_rank:
                    if len(sources) != 1:
                        # Code should never reach here
                        raise ValueError("Something is wrong...")

                    rank = root_name + str(0)
                    self.nodes[rank] = RankNode(rank, 0)
                    self.graph.add_edge(sources[0], self.nodes[rank])


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
