#!/bin/bash
# file name: check.sh

# Type Checking
mypy teaal

# Auto-Formatting
autopep8 -iraa teaal/
autopep8 -iraa tests/

# Remove the newlines
for FILE in $(ls tests/integration | grep "\.py")
do
    perl -pi -e 'chomp if eof' tests/integration/$FILE
done

# Run the corrected files
python -m pytest --cov=teaal --cov-report term-missing tests/parse tests/hifiber tests/ir/test_component.py tests/ir/test_coord_math.py tests/ir/test_equation.py tests/ir/test_flow_nodes.py tests/ir/test_iter_graph.py tests/ir/test_level.py tests/ir/test_loop_order.py tests/ir/test_node.py tests/ir/test_part_nodes.py tests/ir/test_partitioning.py tests/ir/test_program.py tests/ir/test_spacetime.py tests/ir/test_tensor.py tests/trans/test_canvas.py tests/trans/test_coord_access.py tests/trans/test_equation.py tests/trans/test_footer.py tests/trans/test_graphics.py tests/trans/test_header.py tests/trans/test_partitioner.py tests/trans/test_utils.py tests/ir/test_hardware.py tests/ir/test_metrics.py

# python -m pytest tests/ir/test_flow_graph.py
# python -m pytest tests/trans/test_collector.py
# python -m pytest tests/trans/test_hifiber.py
