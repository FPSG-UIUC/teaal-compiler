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
from sympy import Basic, Symbol
from typing import Any, Dict, Iterable, List, Optional, Tuple

from teaal.ir.tensor import Tensor
from teaal.parse.utils import ParseUtils


class Partitioning:
    """
    An abstract representation of the partitioning information
    """

    def __init__(self,
                 partitioning: Dict[str, List[Tree]],
                 ranks: Iterable[str],
                 eqn_exprs: Dict[Symbol, Basic]) -> None:
        """
        Create a new representation of the partitioning information
        """
        self.eqn_exprs = eqn_exprs

        # Filter the partitioning information into the ranks that can
        # be partitioned statically vs dynamically
        self.dyn_parts = {}
        self.static_parts = {}

        for rank, parts in partitioning.items():

            # Continue if this rank is not actually partitioned
            if not parts:
                continue

            if Partitioning.__nway_after_dyn(parts):
                raise ValueError(
                    "N-way partitioning after dynamic partitioning on rank " + rank)

            # Add the partitioning specification to the appropriate dictionary
            if Partitioning.__is_static(parts[0]):
                self.static_parts[rank] = parts
            else:
                self.dyn_parts[rank] = parts

        self.all_parts = {**self.static_parts, **self.dyn_parts}

        # For all names of ranks, e.g., M1 or K2I, save the root name
        # of the rank so we can recover it later
        self.root_names = {}
        for rank in ranks:
            self.root_names[rank] = rank

        for part_rank in self.all_parts.keys():
            for rank in self.__part_to_tensor_rank(part_rank, ranks):
                self.root_names.update(
                    {inter: rank for inter in self.get_intermediates(rank)})
                self.root_names.update({rank + "0": rank})

            self.root_names.update(
                {final: part_rank for final in self.partition_names(part_rank, True)})

        # All possible rank IDs to the final rank ID
        self.final_rank_id = {}

        # Unpartitioned ranks will not change
        for rank in ranks:
            self.final_rank_id[rank] = rank

        # Partitioned ranks may change
        for part_rank in self.all_parts.keys():
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
        init_ranks = [rank for rank in self.all_parts.keys()]
        for rank in init_ranks:
            for i, int_ in enumerate(self.get_intermediates(rank)):
                i = int(int_[len(rank):-1])
                self.dyn_parts[int_] = self.all_parts[rank][-i:]
                self.all_parts[int_] = self.all_parts[rank][-i:]

    def get_all_parts(self) -> Dict[str, List[Tree]]:
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
        if self.__tensor_to_part_rank(rank.upper(), self.dyn_parts.keys()):
            return rank + "0"
        return rank

    def get_dyn_parts(self) -> Dict[str, List[Tree]]:
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
        part_ranks = self.__tensor_to_part_rank(rank, self.all_parts.keys())
        part_rank = Partitioning.__single_part_rank(rank, part_ranks)

        intermediates = []
        for i, part in enumerate(self.all_parts[part_rank][1:]):
            num = len(self.all_parts[part_rank]) - i - 1
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
        return self.root_names[rank]

    def get_static_parts(self) -> Dict[str, List[Tree]]:
        """
        Get the partitioning information for all statically partitioned
        ranks
        """
        return self.static_parts

    def get_tensor_spec(self, tensor_ranks: Iterable[str],
                        part_ranks: Iterable[str]) -> Dict[str, List[Tree]]:
        """
        Get the partitioning for a specific tensor's ranks
        """
        partitioning = {}
        for rank, part in self.all_parts.items():
            if rank in part_ranks and self.__part_to_tensor_rank(
                    rank, tensor_ranks):
                partitioning[rank] = part
        return partitioning

    def partition_names(self, rank: str, all_: bool) -> List[str]:
        """
        Get the list of names that this rank will be partitioned into
        """
        part_ranks = self.__tensor_to_part_rank(rank, self.all_parts.keys())
        part_rank = Partitioning.__single_part_rank(rank, part_ranks)

        parts = self.all_parts[part_rank]
        # Return all final rank names
        if all_:
            parted_ = [part_rank + str(j) for j in range(len(parts) + 1)]
            # The bottom rank is named with the original rank name
            parted_[0] = rank + "0"
            return parted_

        part_root = self.root_names[part_rank]
        names = [part_root + str(len(parts))]

        root = self.root_names[rank]

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
        part_ranks = self.__tensor_to_part_rank(rank, self.all_parts.keys())
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

        for rank in tensor.get_ranks():
            # Check if there is anything to partition
            part_ranks = self.__tensor_to_part_rank(rank, all_parts.keys())
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
