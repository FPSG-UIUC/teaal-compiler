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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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

    def get_dyn_rank(self, rank: str) -> str:
        """
        Convert from a (potentially) static rank name to the corresponding
        dynamic rank name

        Used for the spacetime stamp of dynamically partitioned tensors
        """
        if self.is_flattened(rank):
            raise ValueError(
                "Should never be used for flattened ranks, used on rank " + rank)

        if self.__tensor_to_part_rank(rank, self.dyn_parts):
            return rank + "0"
        return rank

    def get_dyn_parts(self) -> Set[Tuple[str, ...]]:
        """
        Get the partitioning information for all dynamically partitioned
        ranks
        """
        return self.dyn_parts

    def get_final_rank_id(self, tensor: Tensor, rank: str) -> str:
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
                    if self.get_root_name(rank) not in tensor.get_init_ranks():
                        comp = min
            else:
                node = comp(
                    succ, key=lambda n: self.graph.nodes[n]["priority"])

            succ = list(self.graph.successors(node))

        return node.get_rank()

    def get_intermediates(self, tensor: Tensor, rank: str) -> List[str]:
        """
        Get the names of all intermediate ranks (e.g., K2I)
        """
        # TODO allow for flattened ranks
        intermediates: List[str] = []
        node = None
        succ = list(self.graph.successors(RankNode(rank)))
        while succ:
            if isinstance(node, RankNode):
                intermediates.append(node.get_rank())

            if isinstance(succ[0], FlattenNode):
                assert len(succ) == 1
                node = succ[0]

                # Stop if not all flattened ranks are included in the tensor
                for rank in node.get_ranks():
                    if self.get_root_name(rank) not in tensor.get_init_ranks():
                        # Note that the rank feeding to this FlattenNode is not
                        # an intermediate
                        return intermediates[:-1]

            else:
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
            node = RankNode(Partitioning.__best_match(rank, preds))
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

    def get_tensor_spec(self, tensor_ranks: Iterable[str],
                        part_ranks: Iterable[Tuple[str, ...]]) -> Dict[Tuple[str, ...], List[Tree]]:
        """
        Get the partitioning for a specific tensor's ranks
        """
        partitioning: Dict[Tuple[str, ...], List[Tree]] = {}
        used_pranks = []
        # Separate out the used partitioning ranks
        for pranks in part_ranks:
            tranks = []
            used = False
            for prank in pranks:
                trank = self.__part_to_tensor_rank(prank, tensor_ranks)
                if len(trank) > 1:
                    raise ValueError(
                        "Partitioning rank " +
                        prank +
                        " maps to tensor ranks " +
                        str(trank))

                if trank:
                    tranks.extend(trank)
                    used = True
                else:
                    used = False

            if used:
                used_pranks.append(pranks)

        # Get the specification for each partitioning
        for pranks in used_pranks:
            partitioning[pranks] = []

            # If this is a splitting of a single rank into multiple
            if len(pranks) == 1:
                parent = RankNode(pranks[0])
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
                        partitioning[pranks].append(
                            self.graph.edges[(parent, succ)]["part"])

                    parent = succs[-1]
                    succs = [node for node in self.graph.successors(parent)]

            # Otherwise, this is a flattening of many into one
            else:
                flat_node = FlattenNode(pranks)
                rank_node = RankNode("".join(pranks))
                partitioning[pranks].append(
                    self.graph.edges[(flat_node, rank_node)]["part"])

        return partitioning

    def is_flattened(self, rank: str) -> bool:
        """
        Return true if the rank is the result of a flattening
        """
        node = RankNode(rank)

        if "is_flattened" in self.graph.nodes[node].keys():
            return self.graph.nodes[node]["is_flattened"]

        preds = [n.get_rank() for n in self.graph.predecessors(node)]
        while preds:
            node = RankNode(Partitioning.__best_match(rank, preds))
            preds = []
            for n in self.graph.predecessors(node):
                if isinstance(n, FlattenNode):
                    self.graph.nodes[node]["is_flattened"] = True
                    return True

        self.graph.nodes[node]["is_flattened"] = False
        return False

    def partition_names(self, ranks: List[str], all_: bool) -> List[str]:
        """
        Get the list of names that these ranks will be partitioned into
        """
        frontier = [RankNode(rank) for rank in ranks]
        key = {rank: (i,
                      self.graph.nodes[RankNode(rank)]["priority"]) for i,
               rank in enumerate(ranks)}
        pos = len(ranks)
        visited = set()
        leaves: Set[RankNode] = set()

        while frontier:
            node = frontier.pop()
            visited.add(node)

            succs = list(self.graph.successors(node))
            is_leaf = True
            for i in range(len(succs)):
                succ = succs[i]
                if isinstance(succ, RankNode):
                    is_leaf = False
                    key[succ.get_rank()] = (key[node.get_rank()][0],
                                            self.graph.nodes[succ]["priority"])
                    continue

                # Otherwise, we have a FlattenNode
                preds = list(self.graph.predecessors(succ))

                # Ensure that we have visited all input ranks
                if not all(pred in visited for pred in preds):
                    continue

                is_leaf = False

                # Mark all predecessors as no longer leaves
                for pred in preds:
                    if pred in leaves:
                        leaves.remove(pred)

                # Get the rank node
                succ = next(self.graph.successors(succ))
                succs[i] = succ

                # Update the key (position of this rank in the final list)
                key[succ.get_rank()] = (pos, self.graph.nodes[succ]["priority"])
                pos += 1

            if is_leaf:
                leaves.add(node)
            elif all_:
                frontier.extend(succs)
            else:
                leaves.update(succs)

        names = [leaf.get_rank() for leaf in leaves]
        names.sort(key=lambda n: key[n])

        return names

    def partition_rank(self, rank: str) -> Optional[str]:
        """
        Get the name of the corresponding partitioned rank, should one exist
        """
        # TODO allow for flattened ranks
        part_ranks = self.__tensor_to_part_rank(rank, self.all_parts)
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
        tensor_ranks = tensor.get_ranks().copy()

        # TODO allow for flattened ranks
        for rank in tensor.get_ranks():
            # Check if there is anything to partition
            part_ranks = self.__tensor_to_part_rank(rank, self.all_parts)
            if not part_ranks:
                continue

            # Check if we want to partition it
            part_rank = Partitioning.__single_part_rank(rank, part_ranks)
            if rank in ranks or part_rank in ranks:
                # Remove the old rank
                i = tensor_ranks.index(rank)
                tensor_ranks.pop(i)

                # Insert the new ranks
                new_ranks = self.partition_names([rank], all_)
                for new_rank in new_ranks:
                    tensor_ranks.insert(i, new_rank)

        return tensor_ranks

    def __add_or_update_priority(self, rank: str, priority: float) -> None:
        """
        Add the node if it does not exist, or update its priority
        """
        node = RankNode(rank)
        if node not in self.graph.nodes:
            self.graph.add_node(node, priority=priority)
        else:
            self.graph.nodes[node]["priority"] = priority

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
                           orig_ranks: Iterable[str]) -> None:
        """
        Build the graph of how the partitioning information is related
        """

        self.graph = nx.DiGraph()
        ranks = set(orig_ranks)

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
        parted_by: Dict[str, List[str]] = {}
        roots: Dict[str, List[str]] = {}
        edge: Tuple[PartitioningNode, PartitioningNode]
        all_parts = {
            tuple(
                str(child) for child in ranks_tree.children): parts for ranks_tree,
            parts in partitioning.items()}
        for part_ranks, parts in all_parts.items():
            if Partitioning.__nway_after_dyn(parts):
                raise ValueError(
                    "N-way partitioning after dynamic partitioning on rank(s) " +
                    str(part_ranks))

            self.__check_flatten(part_ranks, all_parts, ranks, orig_ranks)

            # If we are flattening, add a flattening node to combine them
            if len(part_ranks) > 1:
                flattened_rank = "".join(part_ranks)
                self.__add_or_update_priority(flattened_rank, 0)

                flatten_node = FlattenNode(part_ranks)
                edge = (flatten_node, RankNode(flattened_rank))
                self.graph.add_edge(
                    *edge, part=parts[0], part_ranks=part_ranks)

                for part_rank in part_ranks:
                    edge = (RankNode(part_rank), flatten_node)
                    self.graph.add_edge(
                        *edge, part=parts[0], part_ranks=part_ranks)

                ranks.add(flattened_rank)
                self.eqn_exprs[Symbol(flattened_rank.lower())] = Symbol(
                    flattened_rank.lower())

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
            sources = [RankNode(root_name) for root_name in roots[part_rank]]
            # Add edges
            for i, part in enumerate(parts):
                j = len(parts) - i
                if Partitioning.__is_static(part):
                    if part_rank not in orig_ranks:
                        raise ValueError(
                            "Shape-based partitioning found on rank " +
                            part_rank +
                            " after flattening")

                    rank = part_rank + str(j)
                    self.__add_or_update_priority(rank, j)

                    for source in sources:
                        edge = (source, RankNode(rank))
                        self.graph.add_edge(
                            *edge, part=part, part_ranks=part_ranks)

                    continue

                # Else dynamic partitioning
                for k in range(len(sources)):
                    # If this is not the first partition, we need an
                    # explicit intermediate
                    if i > 0:
                        int_rank = roots[part_rank][k] + str(j) + "I"
                        self.__add_or_update_priority(int_rank, j)

                        edge = (sources[k], RankNode(int_rank))
                        self.graph.add_edge(
                            *edge, part=parts[i - 1], part_ranks=part_ranks)

                        sources[k] = RankNode(int_rank)

                    rank = part_rank + str(j)
                    self.__add_or_update_priority(rank, j)

                    edge = (sources[k], RankNode(rank))
                    self.graph.add_edge(
                        *edge, part=part, part_ranks=part_ranks)

            # Add the bottom rank if needed
            if len(parts) > 0:
                for root_name, source in zip(roots[part_rank], sources):
                    rank = root_name + str(0)
                    self.__add_or_update_priority(rank, 0)

                    edge = (source, RankNode(rank))
                    self.graph.add_edge(
                        *edge, part=parts[-1], part_ranks=part_ranks)

        # Ensure that each rank is only partitioned with one set of
        # partitioning information
        for rank, partitioners in parted_by.items():
            Partitioning.__single_part_rank(rank, partitioners)

    def __check_flatten(self, part_ranks: Tuple[str, ...], all_parts: Dict[Tuple[str, ...],
                        List[Tree]], all_ranks: Iterable[str], orig_ranks: Iterable[str]) -> None:
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
            if len(self.__part_to_tensor_rank(rank, all_ranks)) > 1:
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

            if rank not in orig_ranks:
                if rank in all_ranks:
                    raise ValueError(
                        "Cannot flatten rank " +
                        rank +
                        " because it is a flattened rank")

                if rank[:-1] not in orig_ranks or rank[-1] != "0":
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
            part_ranks: Iterable[Tuple[str, ...]]) -> List[str]:
        """
        Return a partition rank if one corresponds to the given tensor rank
        """
        tensor_root = self.get_root_name(tensor_rank).lower()
        tensor_suffix = tensor_rank[len(tensor_root):]

        atoms = self.eqn_exprs[Symbol(tensor_root)].atoms(Symbol)
        eqn_ranks = {str(atom).upper() + tensor_suffix for atom in atoms}

        matches = []
        for eqn_rank in eqn_ranks:
            # TODO: Allow for flattening
            if (eqn_rank,) in part_ranks:
                matches.append(eqn_rank)

        return matches
