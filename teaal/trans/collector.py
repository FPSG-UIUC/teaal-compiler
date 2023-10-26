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

Translate the metrics collection
"""

from teaal.hifiber import *
from teaal.ir.component import *
from teaal.ir.fusion import Fusion
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.trans.utils import TransUtils


class Collector:
    """
    Translate the metrics collection
    """

    def __init__(
            self,
            program: Program,
            metrics: Metrics,
            fusion: Fusion) -> None:
        """
        Construct a collector object
        """
        self.program = program
        self.metrics = metrics
        self.fusion = fusion

        # tree_traces: Optional[Dict[rank, Dict[is_read, Set[tensor]]]]
        self.tree_traces: Optional[Dict[str, Dict[bool, Set[str]]]] = None

    def create_component(self, component: Component, rank: str) -> Statement:
        """
        Create a component to track metrics
        """
        name = component.get_name()
        if isinstance(component, LeaderFollowerComponent):
            constructor = "LeaderFollowerIntersector"
        elif isinstance(component, SkipAheadComponent):
            constructor = "SkipAheadIntersector"
        elif isinstance(component, TwoFingerComponent):
            constructor = "TwoFingerIntersector"
        else:
            raise ValueError(
                "Unable to create consumable metrics component for " +
                name + " of type " + type(component).__name__)

        return SAssign(AVar(name + "_" + rank), EFunc(constructor, []))

    def consume_traces(self, component: str, rank: str) -> Statement:
        """
        Consume the traces to track this component
        """
        component_ir = self.metrics.get_hardware().get_component(component)

        if isinstance(component_ir, IntersectorComponent):
            tracker_name = EVar(component + "_" + rank)
            traces = self.metrics.get_coiter_traces(component, rank)
            consume_args = [[AJust(EString(rank)),
                             AJust(EString(trace))] for trace in traces]
            args = [AJust(EMethod(EVar("Metrics"), "consumeTrace", arg))
                    for arg in consume_args]
            return SExpr(EMethod(tracker_name, "addTraces", args))

        else:
            raise ValueError(
                "Unable to consume traces for component " +
                component +
                " of type " +
                type(component_ir).__name__)

    def dump(self) -> Statement:
        """
        Dump metrics information
        """
        block = SBlock([])
        # If this is the first time, create a dictionary to store all
        # of the metrics information
        if self.program.get_einsum_ind() == 0:
            block.add(SAssign(AVar("metrics"), EDict({})))

        einsum = self.program.get_equation().get_output().root_name()
        block.add(
            SAssign(
                AAccess(
                    EVar("metrics"), EString(einsum)), EDict(
                    {})))

        # Create the formats
        block.add(self.__build_formats())

        # Track the traffic
        block.add(self.__build_traffic())

        # Track the merges
        block.add(self.__build_merges())

        # Track the compute
        block.add(self.__build_compute())

        # Track the intersections
        block.add(self.__build_intersections())

        # Track the sequences
        block.add(self.__build_sequencers())

        # Add the final execution time modeling
        num_einsums = len(self.program.get_all_einsums())
        if self.program.get_einsum_ind() + 1 == num_einsums:
            block.add(self.__build_time())

        return block

    @staticmethod
    def end() -> Statement:
        """
        End metrics collection
        """
        return SExpr(EMethod(EVar("Metrics"), "endCollect", []))

    def make_body(self) -> Statement:
        """
        Make the body of the loop
        """
        return self.__make_iter_num("body")

    def make_loop_footer(self, rank: str) -> Statement:
        """
        Make a footer for the loop
        """
        block = SBlock([])

        if self.tree_traces is None:
            raise ValueError(
                "Unconfigured collector. Make sure to first call start()")

        # Collect the iteration number if necessary
        block.add(self.__make_iter_num(rank))

        # Consume a trace if necessary
        coiter = self.metrics.get_coiter(rank)
        if coiter is not None:
            block.add(self.consume_traces(coiter.get_name(), rank))

        # Eagerly store subtrees as necessary
        for tensor in self.tree_traces[rank][False]:
            block.add(self.trace_tree(tensor, rank, False))

        return block

    def make_loop_header(self, rank: str) -> Statement:
        """
        Make a header for a loop
        """
        block = SBlock([])

        if self.tree_traces is None:
            raise ValueError(
                "Unconfigured collector. Make sure to first call start()")

        loop_ranks = ["root"] + self.program.get_loop_order().get_ranks()
        i = loop_ranks.index(rank)

        # Save the set of subtrees already eagerly loaded
        eager_evicts = self.metrics.get_eager_evicts(loop_ranks[i - 1])
        for tensor, root in eager_evicts:
            tracker = "eager_" + tensor.lower() + "_" + root.lower() + "_read"
            block.add(SAssign(AVar(tracker), EFunc("set", ())))

        # Eagerly load new subtrees as necessary
        for tensor in self.tree_traces[rank][True]:
            block.add(self.trace_tree(tensor, rank, True))

        return block

    def register_ranks(self) -> Statement:
        """
        Register the given ranks
        """
        block = SBlock([])
        for rank in self.program.get_loop_order().get_ranks():
            block.add(
                SExpr(
                    EMethod(
                        EVar("Metrics"), "registerRank", [
                            AJust(
                                EString(rank))])))

        return block

    def set_collecting(
            self,
            tensor: Optional[str],
            rank: str,
            type_: str,
            consumable: bool,
            is_read_trace: bool) -> Statement:
        """
        Collect the statistics about a tensor
        """
        block = SBlock([])
        if tensor is None:
            if type_ != "iter":
                raise ValueError(
                    "Tensor must be specified for trace type " + type_)
            trace = "iter"

        elif type_ == "fiber":
            trace = self.metrics.get_fiber_trace(tensor, rank, is_read_trace)

        # Type is an eager rank
        else:
            trace = "eager_" + tensor.lower() + "_" + type_.lower()
            if is_read_trace:
                trace += "_read"
            else:
                trace += "_write"

                # We want to collect the iteration number for the last loop
                # rank
                output = self.program.get_equation().get_tensor(tensor)
                final_tensor = Tensor(
                    output.root_name(), output.get_init_ranks())
                self.program.apply_all_partitioning(final_tensor)
                self.program.get_loop_order().apply(final_tensor)

                iter_var = final_tensor.get_ranks()[-1].lower() + "_iter_num"
                # TODO: Add a separate None type
                block.add(SAssign(AVar(iter_var), EVar("None")))

        args: List[Argument] = [
            AJust(
                EString(rank)), AParam(
                "type_", EString(trace)), AParam(
                "consumable", EBool(consumable))]

        block.add(SExpr(EMethod(EVar("Metrics"), "trace", args)))
        return block

    def start(self) -> Statement:
        """
        Start metrics collection
        """
        block = SBlock([])

        einsum = self.program.get_equation().get_output().root_name()
        prefix = EString(self.metrics.get_hardware().get_prefix(einsum))
        call = EMethod(EVar("Metrics"), "beginCollect", [AJust(prefix)])

        block.add(SExpr(call))

        block.add(self.__build_components())

        block.add(self.__build_match_ranks())

        stmt, register = self.__build_trace_ranks()
        block.add(stmt)

        if register:
            block.add(self.register_ranks())

        return block

    def trace_tree(
            self,
            tensor: str,
            rank: str,
            is_read_trace: bool) -> Statement:
        """
        Trace a subtree under the fiber specified
        """
        loop_ranks = self.program.get_loop_order().get_ranks()
        i = loop_ranks.index(rank)
        output = self.program.get_equation().get_output()

        # If this is a write trace and this is the current outer-most loop,
        # then just use the outermost loop of the tensor
        if not is_read_trace and i == 0:
            rank = output.get_ranks()[0]

        # Otherwise, get the rank of the current fiber
        elif not is_read_trace and i > 0:
            output_ranks = output.get_ranks()
            rank = output_ranks[output_ranks.index(loop_ranks[i - 1]) + 1]

        fiber = tensor.lower() + "_" + rank.lower()

        trace = "eager_" + fiber
        if is_read_trace:
            trace += "_read"
        else:
            trace += "_write"

        args: List[Argument] = [AJust(EString(trace))]
        if not is_read_trace:
            # We want to use the iteration number for the last loop rank
            final_tensor = Tensor(output.root_name(), output.get_init_ranks())
            self.program.apply_all_partitioning(final_tensor)
            self.program.get_loop_order().apply(final_tensor)

            iter_var = final_tensor.get_ranks()[-1].lower() + "_iter_num"
            args.append(AParam("iteration_num", EVar(iter_var)))

        trace_stmt = SExpr(EMethod(EVar(fiber), "trace", args))
        if not is_read_trace:
            return trace_stmt

        # If read, only read the first time
        tensor_ir = self.program.get_equation().get_tensor(tensor)

        get_final = self.program.get_partitioning().get_final_rank_id
        evict_rank = self.metrics.get_eager_evict_on(tensor, rank)[-1]
        er_ind = loop_ranks.index(get_final([evict_rank], evict_rank))
        tree_ind = loop_ranks.index(get_final([rank], rank))

        key = []
        for loop_rank in loop_ranks[er_ind + 1:tree_ind]:
            if loop_rank in tensor_ir.get_ranks():
                key.append(EVar(loop_rank.lower()))
        key_tuple = ETuple(tuple(key))

        cond = EBinOp(key_tuple, ONotIn(), EVar(trace))
        add_key = SExpr(EMethod(EVar(trace), "add", [AJust(key_tuple)]))
        return SIf((cond, SBlock([add_key, trace_stmt])), [], None)

    def __add_collection(self,
                         trace: Tuple[Optional[str],
                                      str,
                                      str,
                                      bool,
                                      bool],
                         traces: Set[Tuple[Optional[str],
                                           str,
                                           str,
                                           bool,
                                           bool]]) -> Statement:
        """
        Add a collection and update the set of traces
        """
        if trace not in traces:
            traces.add(trace)
            return self.set_collecting(*trace)

        return SBlock([])

    def __get_trace(self, binding: dict,
                    is_read: bool) -> Tuple[str, Statement]:
        """
        Get the (trace, HiFiber to produce the trace)
        """
        einsum = self.program.get_equation().get_output().root_name()
        prefix = self.metrics.get_hardware().get_prefix(einsum) + \
            "-" + binding["rank"] + "-"

        block = SBlock([])
        if "style" in binding and binding["style"] == "eager":
            trace_fn = prefix + "eager_" + \
                binding["tensor"].lower() + "_" + binding["root"].lower()
            if is_read:
                trace_fn += "_read"
            else:
                trace_fn += "_write"
            trace_fn += ".csv"

        # Otherwise binding is lazy
        else:
            fiber_trace = self.metrics.get_fiber_trace(
                binding["tensor"], binding["rank"], is_read)

            if binding["type"] == "payload" and fiber_trace != "iter" and \
                    fiber_trace[:11] != "get_payload":
                input_fn = prefix + fiber_trace + ".csv"
                filter_fn = prefix + "iter.csv"
                trace_fn = prefix + fiber_trace + "_payload.csv"

                args = [AJust(EString(fn))
                        for fn in [input_fn, filter_fn, trace_fn]]
                block.add(SExpr(EMethod(EVar("Traffic"), "filterTrace", args)))

            else:
                trace_fn = prefix + fiber_trace + ".csv"

        return trace_fn, block

    def __build_components(self) -> Statement:
        """
        Build the creation of any necessary hardware components
        """
        block = SBlock([])
        einsum = self.program.get_equation().get_output().root_name()

        for component in self.metrics.get_hardware().get_components(einsum,
                                                                    IntersectorComponent):
            name = component.get_name()

            for binding in component.get_bindings()[einsum]:
                block.add(self.create_component(component, binding["rank"]))

        return block

    def __build_compute(self) -> Statement:
        """
        Add the code to count compute operations
        """
        block = SBlock([])
        einsum = self.program.get_equation().get_output().root_name()

        metrics_einsum = EAccess(EVar("metrics"), EString(einsum))
        metrics_dump = EAccess(
            EMethod(
                EVar("Metrics"),
                "dump",
                []),
            EString("Compute"))
        for fu in self.metrics.get_hardware().get_components(einsum, ComputeComponent):
            block.add(
                SAssign(
                    AAccess(
                        metrics_einsum, EString(
                            fu.get_name())), EDict(
                        {})))

            metrics_fu = EAccess(metrics_einsum, EString(fu.get_name()))
            ops = []
            for binding in fu.get_bindings()[einsum]:
                op = binding["op"]
                ops.append(EString(op))
                block.add(
                    SAssign(
                        AAccess(metrics_fu, EString(op)),
                        EAccess(metrics_dump, EString("payload_" + op))))

            # TODO: Handle multi-op functional units
            assert len(ops) == 1

            # op_freq = cycles / s * ops / cycle
            op_freq = self.metrics.get_hardware().get_frequency(einsum) * \
                fu.get_num_instances()
            time = EBinOp(EAccess(metrics_fu, ops[0]), ODiv(), EInt(op_freq))

            metrics_time = AAccess(metrics_fu, EString("time"))
            block.add(SAssign(metrics_time, time))
            self.fusion.add_component(einsum, fu.get_name())

        return block

    def __build_formats(self) -> Statement:
        """
        Add the code to build the formats dictionary
        """
        formats_dict: Dict[Expression, Expression] = {}
        part_ir = self.program.get_partitioning()
        for tensor, format_ in self.metrics.get_loop_formats().items():
            loop_format = self.metrics.get_format().get_spec(tensor)[format_]
            rank_order = loop_format["rank-order"]

            # If there is dynamic partitioning applied we cannot use the
            # existing tensor
            build_new = False

            # TODO: This should be in teaal.ir.partitioning
            tensor_ir = self.program.get_equation().get_tensor(tensor)
            old_ranks: List[str] = []
            new_ranks = tensor_ir.get_init_ranks()
            while old_ranks != new_ranks:
                old_ranks = new_ranks
                new_ranks = part_ir.partition_ranks(
                    new_ranks, part_ir.get_static_parts(), False, True)

            for static_rank in new_ranks:
                if (static_rank,) in part_ir.get_dyn_parts():
                    build_new = True
                    break

                if part_ir.is_flattened(static_rank):
                    build_new = True
                    break

            tensor_expr: Expression
            if build_new:
                rank_ids = TransUtils.build_expr(rank_order)

                shape: List[Expression] = []
                for rank in rank_order:
                    if not part_ir.is_flattened(rank):
                        shape.append(EVar(part_ir.get_root_name(rank)))
                        continue

                    unpacked = part_ir.unpack(rank)
                    roots = [part_ir.get_root_name(src) for src in unpacked]
                    rank_shape: Expression = EVar(roots[0])
                    for root in roots[1:]:
                        rank_shape = EBinOp(rank_shape, OMul(), EVar(root))
                    shape.append(rank_shape)

                args = [
                    AParam(
                        "rank_ids", rank_ids), AParam(
                        "shape", EList(shape))]
                tensor_expr = EFunc("Tensor", args)

            else:
                tensor_expr = EVar(
                    tensor + "_" + "".join(rank_order))

            format_yaml = TransUtils.build_expr(loop_format)

            formats_dict[EString(tensor)] = EFunc(
                "Format", [AJust(tensor_expr), AJust(format_yaml)])

        return SAssign(AVar("formats"), EDict(formats_dict))

    def __build_intersections(self) -> Statement:
        """
        Add the code to compute the intersection operations
        """
        block = SBlock([])
        einsum = self.program.get_equation().get_output().root_name()

        metrics_einsum = EAccess(EVar("metrics"), EString(einsum))
        for intersector in self.metrics.get_hardware().get_components(einsum,
                                                                      IntersectorComponent):
            isect_name = intersector.get_name()
            metrics_isect = AAccess(metrics_einsum, EString(isect_name))
            block.add(SAssign(metrics_isect, EInt(0)))

            for binding in intersector.get_bindings()[einsum]:
                isects = EMethod(
                    EVar(
                        isect_name +
                        "_" +
                        binding["rank"]),
                    "getNumIntersects",
                    [])
                block.add(SIAssign(metrics_isect, OAdd(), isects))

            # op_freq = cycles / s * ops / cycle
            op_freq = self.metrics.get_hardware().get_frequency(einsum) * \
                intersector.get_num_instances()
            metrics_isect_expr = EAccess(metrics_einsum, EString(isect_name))
            time = EBinOp(metrics_isect_expr, ODiv(), EInt(op_freq))

            metrics_time = AAccess(metrics_isect_expr, EString("time"))
            block.add(SAssign(metrics_time, time))
            self.fusion.add_component(einsum, intersector.get_name())

        return block

    def __build_match_ranks(self) -> Statement:
        """
        Add the code to match ranks, e.g., if we have flattening
        """
        block = SBlock([])

        part_ir = self.program.get_partitioning()
        for rank in self.program.get_loop_order().get_ranks():
            if not part_ir.is_flattened(rank):
                continue

            unpacked = part_ir.unpack(rank)
            roots = []
            for unpack_rank in unpacked:
                if part_ir.get_final_rank_id(
                        [unpack_rank], unpack_rank) == rank:
                    args = [AJust(EString(rank)), AJust(EString(unpack_rank))]
                    block.add(
                        SExpr(
                            EMethod(
                                EVar("Metrics"),
                                "matchRanks",
                                args)))

                roots.append(EVar(part_ir.get_root_name(unpack_rank)))

            args = [AJust(EString(rank)), AJust(ETuple(tuple(roots)))]
            block.add(SExpr(EMethod(EVar("Metrics"), "associateShape", args)))

        return block

    def __build_merges(self) -> Statement:
        """
        Add the code to compute the merge operations
        """
        block = SBlock([])
        einsum = self.program.get_equation().get_output().root_name()

        metrics_einsum = EAccess(EVar("metrics"), EString(einsum))
        for merger in self.metrics.get_hardware().get_components(einsum, MergerComponent):
            merger_name = merger.get_name()
            block.add(
                SAssign(
                    AAccess(
                        metrics_einsum, EString(merger_name)), EDict(
                        {})))
            metrics_merger = EAccess(metrics_einsum, EString(merger_name))
            tensors = []
            for binding in merger.get_bindings()[einsum]:
                init_ranks = binding["init-ranks"]
                final_ranks = binding["final-ranks"]

                input_ = binding["tensor"] + "_" + "".join(init_ranks)
                tensor_name = EVar(input_)
                tensors.append(tensor_name)

                # TODO: Way more complicated merges are possible than a single
                # swap
                depth = EInt([i == f for i, f in zip(
                    init_ranks, final_ranks)].index(False))

                # TODO: Need to first update the HiFiber to use new merge
                # hardware spec
                radix = TransUtils.build_expr(merger.get_comparator_radix())
                next_latency: Expression
                if merger.get_inputs() < float("inf"):
                    next_latency = EInt(1)
                else:
                    next_latency = EString("N")

                args = [
                    AJust(expr) for expr in [
                        tensor_name,
                        depth,
                        radix,
                        next_latency]]
                swaps_call = EMethod(EVar("Compute"), "numSwaps", args)
                block.add(
                    SAssign(
                        AAccess(
                            metrics_merger,
                            EString(input_)),
                        swaps_call))

            # Compute the time required
            # TODO: Support more than one tensor per merger
            assert len(tensors) == 1

            # op_freq = cycles / s * ops / cycle
            op_freq = self.metrics.get_hardware().get_frequency(einsum) * \
                merger.get_num_instances()
            time = EBinOp(
                EAccess(
                    metrics_merger,
                    tensors[0]),
                ODiv(),
                EInt(op_freq))

            metrics_time = AAccess(metrics_merger, EString("time"))
            block.add(SAssign(metrics_time, time))
            self.fusion.add_component(einsum, merger.get_name())

        return block

    def __build_sequencers(self) -> Statement:
        """
        Add a block to track the sequencers
        """
        block = SBlock([])

        einsum = self.program.get_equation().get_output().root_name()
        metrics_einsum = EAccess(EVar("metrics"), EString(einsum))

        for seq in self.metrics.get_hardware().get_components(einsum, SequencerComponent):
            seq_assn = AAccess(metrics_einsum, EString(seq.get_name()))
            block.add(SAssign(seq_assn, EDict({})))
            seq_expr = EAccess(metrics_einsum, EString(seq.get_name()))

            ranks = []
            for rank in seq.get_ranks(einsum):
                ranks.append(rank)
                trace = self.metrics.get_hardware().get_prefix(einsum) + \
                    "-" + rank + "-iter.csv"
                num_iters = EMethod(
                    EVar("Compute"), "numIters", [
                        AJust(
                            EString(trace))])
                seq_rank = AAccess(seq_expr, EString(rank))
                block.add(SAssign(seq_rank, num_iters))

            # Compute time
            steps: Optional[Expression] = None
            for rank in ranks:
                new_steps = EAccess(seq_expr, EString(rank))
                if steps:
                    steps = EBinOp(steps, OAdd(), new_steps)
                else:
                    steps = new_steps

            assert steps is not None

            op_freq = self.metrics.get_hardware().get_frequency(einsum) * \
                seq.get_num_instances()
            time = EBinOp(EParens(steps), ODiv(), EInt(op_freq))

            metrics_time = AAccess(seq_expr, EString("time"))
            block.add(SAssign(metrics_time, time))
            self.fusion.add_component(einsum, seq.get_name())

        return block

    def __build_time(self) -> Statement:
        """
        Add the code necessary to compute the final execution time
        """
        sblock = SBlock([])

        # Save the Einsum blocks
        metrics = EVar("metrics")
        blocks = TransUtils.build_expr(self.fusion.get_blocks())
        sblock.add(SAssign(AAccess(metrics, EString("blocks")), blocks))

        # Compute the execution time
        time: Optional[Expression] = None
        for block in self.fusion.get_blocks():

            # Collect up the statistics for the block
            component_time: Dict[str, Expression] = {}
            for einsum in block:
                metrics_einsum = EAccess(metrics, EString(einsum))
                for comp in self.fusion.get_components(einsum):
                    new_time = EAccess(
                        EAccess(
                            metrics_einsum,
                            EString(comp)),
                        EString("time"))

                    if comp in component_time:
                        component_time[comp] = EBinOp(
                            component_time[comp], OAdd(), new_time)
                    else:
                        component_time[comp] = new_time

            # Sort components to enable testing
            comps = sorted(component_time.keys())

            # Compute block time by taking the max
            block_time: Expression
            if len(comps) == 0:
                block_time = EInt(0)
            elif len(comps) == 1:
                block_time = component_time[comp]
            else:
                comp_args = [AJust(component_time[comp]) for comp in comps]
                block_time = EFunc("max", comp_args)

            # The execution time is the sum of all of the blocks
            if time:
                time = EBinOp(time, OAdd(), block_time)
            else:
                time = block_time

        assert time is not None

        sblock.add(SAssign(AAccess(metrics, EString("time")), time))

        return sblock

    def __build_trace_ranks(self) -> Tuple[Statement, bool]:
        """
        Add code to trace all necessary ranks
        Returns (new code, need to register ranks explicitly)

        Note: explicit rank registration is necessary if we have eager loading
        of fibers
        """
        block = SBlock([])
        einsum = self.program.get_equation().get_output().root_name()
        loop_order = self.program.get_loop_order().get_ranks()

        traces: Set[Tuple[Optional[str], str, str, bool, bool]] = set()
        trace: Tuple[Optional[str], str, str, bool, bool]

        register = False
        self.tree_traces = {rank: {True: set(), False: set()}
                            for rank in loop_order}
        available = [(rank, self.program.get_partitioning().get_available(rank))
                     for rank in reversed(loop_order)]

        for sequencer in self.metrics.get_hardware().get_components(einsum,
                                                                    SequencerComponent):
            for rank in sequencer.get_ranks(einsum):
                trace = (None, rank, "iter", False, True)
                block.add(self.__add_collection(trace, traces))

        for tensor in self.program.get_equation().get_tensors():
            tensor_name = tensor.root_name()

            # Collect the necessary traces for each tensor
            for rank, type_, consumable in self.metrics.get_collected_tensor_info(
                    tensor_name):

                # If we are collecting the loop's trace
                if type_ == "iter":
                    trace = (None, rank, type_, consumable, True)
                    block.add(self.__add_collection(trace, traces))

                # Otherwise, get the fiber's read (and maybe write)
                else:
                    trace = (tensor_name, rank, type_, consumable, True)
                    block.add(self.__add_collection(trace, traces))

                    if tensor.get_is_output():
                        trace = (tensor_name, rank, type_, consumable, False)
                        block.add(self.__add_collection(trace, traces))

                    # Type is fiber if lazy and root of the eager access if
                    # lazy
                    if type_ != "fiber":

                        # Register the rank order explicitly
                        register = True

                        # Eagerly load a subtree right before the given loop
                        loaded = False
                        for loop_rank, avail in available:
                            if type_ in avail:
                                self.tree_traces[loop_rank][True].add(
                                    tensor_name)
                                loaded = True
                                break
                        assert loaded

                        # Eagerly store a subtree right before we move onto the
                        # next subtree
                        if tensor.get_is_output():
                            final_tensor = Tensor(
                                tensor.root_name(), tensor.get_init_ranks())
                            self.program.apply_all_partitioning(final_tensor)
                            self.program.get_loop_order().apply(final_tensor)

                            i = final_tensor.get_ranks().index(type_)
                            if i == 0:
                                store_rank = loop_order[0]
                            else:
                                one_above_rank = final_tensor.get_ranks()[
                                    i - 1]

                                stored = False
                                for j, (loop_rank, avail) in enumerate(
                                        available):
                                    if one_above_rank in avail:
                                        stored = True
                                        break
                                assert stored

                                # Unreversed index -> len(loop_order) - j - 1
                                # Store rank is one below -> + 1
                                store_rank = loop_order[len(loop_order) - j]

                            # Trace the eager tree
                            self.tree_traces[store_rank][False].add(
                                tensor_name)

        return block, register

    def __build_traffic(self) -> Statement:
        """
        Add the code to compute traffic
        """
        block = SBlock([])
        einsum = self.program.get_equation().get_output().root_name()

        active_bindings: Dict[str, List[dict]] = {}
        # Filter out the bindings to ignore
        for buffer_ in self.metrics.get_hardware().get_components(einsum, BufferComponent):
            active_bindings[buffer_.get_name()] = []
            for binding in buffer_.get_bindings()[einsum]:
                format_ = self.metrics.get_format().get_spec(
                    binding["tensor"])[binding["format"]]
                rank = binding["rank"]
                type_ = binding["type"]

                # First make sure that this binding actually corresponds to
                # traffic
                check_cbits = type_ == "coord" or type_ == "elem"
                check_pbits = type_ == "payload" or type_ == "elem"
                if check_cbits and (
                        "cbits" not in format_[rank] or format_[rank]["cbits"] == 0):
                    # Inconsequential line to make the coverage test go in here
                    x = 1
                    continue
                if check_pbits and (
                        "pbits" not in format_[rank] or format_[rank]["pbits"] == 0):
                    # Inconsequential line to make the coverage test go in here
                    x = 1
                    continue

                active_bindings[buffer_.get_name()].append(binding)

        metrics_einsum = EAccess(EVar("metrics"), EString(einsum))
        traffic_dict: Dict[str, Set[str]] = {}
        for buffer_ in self.metrics.get_hardware().get_components(einsum, BufferComponent):
            bindings = TransUtils.build_expr(
                active_bindings[buffer_.get_name()])
            bindings_var = AVar("bindings")

            block.add(SAssign(bindings_var, bindings))

            # Create the traces for each buffer
            # TODO: What if the binding is for an unswizzled tensor
            traces = {}
            ranks = set()
            for binding in active_bindings[buffer_.get_name()]:
                format_ = self.metrics.get_format().get_spec(
                    binding["tensor"])[binding["format"]]
                rank = binding["rank"]
                ranks.add(rank)
                type_ = binding["type"]

                # Now add the trace
                trace, create_trace = self.__get_trace(binding, True)
                block.add(create_trace)
                traces[(binding["tensor"], rank, type_, "read")] = trace
                tensor_ir = self.program.get_equation(
                ).get_tensor(binding["tensor"])
                if tensor_ir.get_is_output():
                    trace, create_trace = self.__get_trace(binding, False)
                    block.add(create_trace)
                    traces[(binding["tensor"], rank, type_, "write")] = trace

                # Also need to add the evict-on rank to the set of ranks if one
                # exists
                if "evict-on" in binding:
                    ranks.add(binding["evict-on"])

            traces_dict = TransUtils.build_expr(traces)
            block.add(SAssign(AVar("traces"), traces_dict))

            args = [
                AJust(
                    EVar("bindings")),
                AJust(
                    EVar("formats")),
                AJust(
                    EVar("traces")),
                AJust(
                    TransUtils.build_expr(
                        buffer_.get_width() *
                        buffer_.get_depth())),
                AJust(
                    TransUtils.build_expr(
                        buffer_.get_width()))]

            # Match ranks not in the loop order to their corresponding rank in
            # the loop order
            rank_map = {}
            for rank in ranks:
                if rank == "root":
                    continue

                final_rank = self.program.get_partitioning(
                ).get_final_rank_id([rank], rank)
                if final_rank != rank:
                    rank_map[rank] = final_rank

            if rank_map:
                args.append(AJust(TransUtils.build_expr(rank_map)))

            if isinstance(buffer_, BuffetComponent):
                traffic_func = "buffetTraffic"
            # Buffer is a cache
            else:
                traffic_func = "cacheTraffic"

            block.add(
                SAssign(
                    AVar("traffic"),
                    EMethod(
                        EVar("Traffic"),
                        traffic_func,
                        args)))

            # Now add it to the metrics dictionary
            added = set()
            for binding in active_bindings[buffer_.get_name()]:
                tensor = binding["tensor"]
                rank = binding["rank"]
                type_ = binding["type"]

                tensor_ir = self.program.get_equation().get_tensor(tensor)
                src_component = self.metrics.get_source_memory(
                    buffer_.get_name(), tensor, rank, type_)

                if src_component is None:
                    continue

                src = src_component.get_name()

                if src not in traffic_dict:
                    traffic_dict[src] = set()
                    block.add(
                        SAssign(
                            AAccess(
                                metrics_einsum, EString(src)), EDict(
                                {})))

                metrics_src = EAccess(metrics_einsum, EString(src))
                metrics_tensor = EAccess(metrics_src, EString(tensor))
                if tensor not in traffic_dict[src]:
                    traffic_dict[src].add(tensor)
                    block.add(
                        SAssign(
                            AAccess(
                                metrics_src, EString(tensor)), EDict(
                                {})))
                    block.add(
                        SAssign(
                            AAccess(
                                metrics_tensor,
                                EString("read")),
                            EInt(0)))

                    if tensor_ir.get_is_output():
                        block.add(
                            SAssign(
                                AAccess(
                                    metrics_tensor,
                                    EString("write")),
                                EInt(0)))

                if (src, tensor) not in added:
                    traffic_access = EAccess(
                        EAccess(
                            EVar("traffic"),
                            EInt(0)),
                        EString(tensor))
                    block.add(
                        SIAssign(
                            AAccess(
                                metrics_tensor,
                                EString("read")),
                            OAdd(),
                            EAccess(
                                traffic_access,
                                EString("read"))))

                    if tensor_ir.get_is_output():
                        block.add(
                            SIAssign(
                                AAccess(
                                    metrics_tensor, EString("write")),
                                OAdd(),
                                EAccess(traffic_access, EString("write"))))

                    added.add((src, tensor))

        # Compute the time it took to perform this traffic
        for src, tensors in traffic_dict.items():
            bits: Optional[Expression] = None
            metrics_src = EAccess(metrics_einsum, EString(src))

            # Note: not technically necessary, just to make the testing
            # deterministic
            sorted_tensors = sorted(tensors)

            for tensor in sorted_tensors:
                metrics_tensor = EAccess(metrics_src, EString(tensor))
                new_bits: Expression = EAccess(metrics_tensor, EString("read"))

                if tensor == einsum:
                    new_bits = EBinOp(
                        new_bits, OAdd(), EAccess(
                            metrics_tensor, EString("write")))

                if bits:
                    bits = EBinOp(bits, OAdd(), new_bits)
                else:
                    bits = new_bits

            # Should always have at least some traffic (error above if not)
            assert bits is not None
            bits = EParens(bits)

            component = self.metrics.get_hardware().get_component(src)
            assert isinstance(component, MemoryComponent)

            metrics_time = AAccess(metrics_src, EString("time"))
            # Note: the current model assumes perfect load balance
            time = EBinOp(
                bits,
                ODiv(),
                EInt(
                    component.get_bandwidth() *
                    component.get_num_instances()))

            block.add(SAssign(metrics_time, time))
            self.fusion.add_component(einsum, src)

        return block

    def __make_iter_num(self, rank: str) -> Statement:
        """
        Save the iteration number if necessary
        """
        # We don't need the iteration number if we are not doing an eager write
        if not self.metrics.get_eager_write():
            return SBlock([])

        loop_order = self.program.get_loop_order().get_ranks() + ["body"]
        output = self.program.get_equation().get_output()

        # We want to collect the iteration number for the last loop rank
        final_tensor = Tensor(output.root_name(), output.get_init_ranks())
        self.program.apply_all_partitioning(final_tensor)
        self.program.get_loop_order().apply(final_tensor)

        # We don't need the iteration number of this rank if it is the top rank
        # since we can never eager access a 0-tensor
        i = loop_order.index(rank)
        if i == 0:
            return SBlock([])

        # We only want the iteration number of the output's bottom rank
        if loop_order[i - 1] != final_tensor.get_ranks()[-1]:
            return SBlock([])

        iter_var = AVar(final_tensor.get_ranks()[-1].lower() + "_iter_num")
        iter_num = EMethod(EMethod(EVar("Metrics"), "getIter", []), "copy", [])

        return SAssign(iter_var, iter_num)
