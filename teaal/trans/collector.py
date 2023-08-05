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
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.trans.utils import TransUtils


class Collector:
    """
    Translate the metrics collection
    """

    def __init__(self, program: Program, metrics: Metrics) -> None:
        """
        Construct a collector object
        """
        self.program = program
        self.metrics = metrics

    def create_component(self, component: str, rank: str) -> Statement:
        """
        Create a component to track metrics
        """
        component_ir = self.metrics.get_hardware().get_component(component)

        if isinstance(component_ir, LeaderFollowerComponent):
            constructor = "LeaderFollowerIntersector"
        elif isinstance(component_ir, SkipAheadComponent):
            constructor = "SkipAheadIntersector"
        elif isinstance(component_ir, TwoFingerComponent):
            constructor = "TwoFingerIntersector"
        else:
            raise ValueError(
                "Unable to create consumable metrics component for " +
                component +
                " of type " +
                type(component_ir).__name__)

        return SAssign(AVar(component + "_" + rank), EFunc(constructor, []))

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
        block = SBlock([])
        if self.metrics.get_eager_write():
            iter_num = EMethod(
                EMethod(
                    EVar("Metrics"),
                    "getIter",
                    []),
                "copy",
                [])
            block.add(SAssign(AVar("iteration_num"), iter_num))

        return block

    def make_loop_header(self, rank: str) -> Statement:
        """
        Make a header for a loop

        TODO: Swallow other metrics counting into here
        """
        block = SBlock([])

        loop_ranks = ["root"] + self.program.get_loop_order().get_ranks()
        i = loop_ranks.index(rank)

        eager_evicts = self.metrics.get_eager_evicts(loop_ranks[i - 1])
        for tensor, root in eager_evicts:
            tracker = "eager_" + tensor.lower() + "_" + root.lower() + "_read"
            block.add(SAssign(AVar(tracker), EFunc("set", ())))

        return block

    def register_ranks(self, ranks: List[str]) -> Statement:
        """
        Register the given ranks
        """
        block = SBlock([])
        for rank in ranks:
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
                # TODO: Add a separate None type
                block.add(SAssign(AVar("iteration_num"), EVar("None")))

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

        TODO: Maybe the CollectingNodes, RegisterRanksNode, etc. can just be
        swallowed up in this
        """
        einsum = self.program.get_equation().get_output().root_name()
        prefix = EString(self.metrics.get_hardware().get_prefix(einsum))
        call = EMethod(EVar("Metrics"), "beginCollect", [AJust(prefix)])

        return SExpr(call)

    def trace_tree(
            self,
            tensor: str,
            rank: str,
            is_read_trace: bool) -> Statement:
        """
        Trace a subtree under the fiber specified
        """
        fiber = tensor.lower() + "_" + rank.lower()

        trace = "eager_" + fiber
        if is_read_trace:
            trace += "_read"
        else:
            trace += "_write"

        args: List[Argument] = [AJust(EString(trace))]
        if not is_read_trace:
            args.append(AParam("iteration_num", EVar("iteration_num")))

        trace_stmt = SExpr(EMethod(EVar(fiber), "trace", args))
        if not is_read_trace:
            return trace_stmt

        # If read, only read the first time
        loop_ranks = self.program.get_loop_order().get_ranks()
        tensor_ir = self.program.get_equation().get_tensor(tensor)

        evict_rank = self.metrics.get_eager_evict_on(tensor, rank)[-1]
        er_ind = loop_ranks.index(evict_rank)
        tree_ind = loop_ranks.index(rank)

        key = []
        for loop_rank in loop_ranks[er_ind + 1:tree_ind]:
            if loop_rank in tensor_ir.get_ranks():
                key.append(EVar(loop_rank.lower()))
        key_tuple = ETuple(tuple(key))

        cond = EBinOp(key_tuple, ONotIn(), EVar(trace))
        add_key = SExpr(EMethod(EVar(trace), "add", [AJust(key_tuple)]))
        return SIf((cond, SBlock([add_key, trace_stmt])), [], None)

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

            if binding["type"] == "payload":
                input_fn = prefix + fiber_trace + ".csv"
                filter_fn = prefix + "iter.csv"
                trace_fn = prefix + fiber_trace + "_payload.csv"

                args = [AJust(EString(fn))
                        for fn in [input_fn, filter_fn, trace_fn]]
                block.add(SExpr(EMethod(EVar("Traffic"), "filterTrace", args)))

            else:
                trace_fn = prefix + fiber_trace + ".csv"

        return trace_fn, block

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
            for binding in fu.get_bindings()[einsum]:
                op = binding["op"]
                block.add(
                    SAssign(
                        AAccess(metrics_fu, EString(op)),
                        EAccess(metrics_dump, EString("payload_" + op))))

        return block

    def __build_formats(self) -> Statement:
        """
        Add the code to build the formats dictionary
        """
        formats_dict: Dict[Expression, Expression] = {}
        for tensor, format_ in self.metrics.get_loop_formats().items():
            loop_format = self.metrics.get_format().get_spec(tensor)[format_]
            tensor_var = EVar(
                tensor + "_" + "".join(loop_format["rank-order"]))

            format_yaml = TransUtils.build_expr(loop_format)

            formats_dict[EString(tensor)] = EFunc(
                "Format", [AJust(tensor_var), AJust(format_yaml)])

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

            for binding in merger.get_bindings()[einsum]:
                init_ranks = binding["init-ranks"]
                final_ranks = binding["final-ranks"]

                input_ = binding["tensor"] + "_" + "".join(init_ranks)
                tensor_name = EVar(input_)
                # TODO: Way more complicated merges are possible than a single
                # swap
                depth = EInt([i == f for i, f in zip(
                    init_ranks, final_ranks)].index(False))

                # TODO: This is very bad; Need to first update the HiFiber
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

        return block

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
            for binding in active_bindings[buffer_.get_name()]:
                format_ = self.metrics.get_format().get_spec(
                    binding["tensor"])[binding["format"]]
                rank = binding["rank"]
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

        return block
