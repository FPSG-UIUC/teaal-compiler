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
python -m pytest --cov=teaal --cov-report term-missing tests/parse tests/hifiber tests/ir tests/trans/test_canvas.py tests/trans/test_coord_access.py tests/trans/test_equation.py tests/trans/test_footer.py tests/trans/test_graphics.py tests/trans/test_header.py tests/trans/test_partitioner.py tests/trans/test_utils.py tests/trans/test_collector.py
# python -m pytest tests/trans/test_hifiber.py
