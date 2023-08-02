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
        formats_dict: Dict[Expression, Expression] = {}
        for tensor, format_ in self.metrics.get_loop_formats().items():
            loop_format = self.metrics.get_format().get_spec(tensor)[format_]
            tensor_var = EVar(
                tensor +
                "_" +
                "".join(
                    loop_format["rank-order"]))

            format_yaml = TransUtils.build_expr(loop_format)

            formats_dict[EString(tensor)] = EFunc(
                "Format", [AJust(tensor_var), AJust(format_yaml)])

        block.add(SAssign(AVar("formats"), EDict(formats_dict)))

        # Create the bindings
        metrics_einsum = EAccess(EVar("metrics"), EString(einsum))
        metrics_dict: Dict[str, Set[str]] = {}
        for buffer_ in self.metrics.get_hardware().get_components(einsum, BufferComponent):
            bindings = TransUtils.build_expr(buffer_.get_bindings()[einsum])
            bindings_var = AVar("bindings")

            block.add(SAssign(bindings_var, bindings))

            # Create the traces for each buffer
            # TODO: What if the binding is for an unswizzled tensor
            traces = {}
            for binding in buffer_.get_bindings()[einsum]:
                trace, create_trace = self.__get_trace(binding, True)
                block.add(create_trace)
                traces[(binding["tensor"], binding["rank"],
                        binding["type"], "read")] = trace

                # We also need a write trace for the output
                output = self.program.get_equation().get_output().root_name()
                if binding["tensor"] == output:
                    trace, create_trace = self.__get_trace(binding, False)
                    block.add(create_trace)
                    traces[(binding["tensor"], binding["rank"],
                            binding["type"], "write")] = trace

            traces_dict = TransUtils.build_expr(traces)
            block.add(SAssign(AVar("traces"), traces_dict))

            python_args = [
                "bindings",
                "formats",
                "traces",
                buffer_.get_width() *
                buffer_.get_depth(),
                buffer_.get_width()]
            args = [AJust(TransUtils.build_expr(arg)) for arg in python_args]

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
            for binding in buffer_.get_bindings()[einsum]:
                tensor = binding["tensor"]
                tensor_ir = self.program.get_equation().get_tensor(tensor)
                src_component = self.metrics.get_source_memory(
                    buffer_.get_name(), tensor, binding["rank"], binding["type"])

                if src_component is None:
                    continue

                src = src_component.get_name()

                if src not in metrics_dict:
                    metrics_dict[src] = set()
                    block.add(
                        SAssign(
                            AAccess(
                                metrics_einsum, EString(src)), EDict(
                                {})))

                metrics_src = EAccess(metrics_einsum, EString(src))
                metrics_tensor = EAccess(metrics_src, EString(tensor))
                if tensor not in metrics_dict[src]:
                    metrics_dict[src].add(tensor)
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
                    traffic_access = EAccess(EVar("traffic"), EString(tensor))
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
                                    metrics_tensor, EString("write")), OAdd(), EAccess(
                                    traffic_access, EString("write"))))

                    added.add((src, tensor))

        return block

    @staticmethod
    def end() -> Statement:
        """
        End metrics collection
        """
        return SExpr(EMethod(EVar("Metrics"), "endCollect", []))

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
        if tensor is None:
            if type_ != "iter":
                raise ValueError(
                    "Tensor must be specified for trace type " + type_)
            trace = "iter"

        # type == "fiber"
        else:
            if type_ != "fiber":
                raise ValueError(
                    "Unable to collect " +
                    type_ +
                    " traces for a specific tensor " +
                    tensor)
            trace = self.metrics.get_fiber_trace(tensor, rank, is_read_trace)

        args: List[Argument] = [
            AJust(
                EString(rank)), AParam(
                "type_", EString(trace)), AParam(
                "consumable", EBool(consumable))]

        return SExpr(EMethod(EVar("Metrics"), "trace", args))

    def start(self) -> Statement:
        """
        Start metrics collection
        """
        loop_order = self.program.get_loop_order()
        order = [EString(rank) for rank in loop_order.get_ranks()]
        call = EMethod(EVar("Metrics"), "beginCollect", [AJust(EList(order))])

        return SExpr(call)

    def __get_trace(self, binding: dict,
                    is_read: bool) -> Tuple[str, Statement]:
        """
        Get the (trace, HiFiber to produce the trace)
        """
        einsum = self.program.get_equation().get_output().root_name()
        prefix = self.metrics.get_hardware().get_prefix(einsum) + \
            "-" + binding["rank"] + "-"
        fiber_trace = self.metrics.get_fiber_trace(
            binding["tensor"], binding["rank"], is_read)

        block = SBlock([])
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
