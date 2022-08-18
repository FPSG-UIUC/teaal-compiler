#!/bin/bash
# file name: check.sh

# Type Checking
mypy es2hfa

# Auto-Formatting
autopep8 -iraa es2hfa/
autopep8 -iraa tests/

# Testing
for FILE in $(ls tests/integration | grep .hfa)
do
    perl -pi -e 'chomp if eof' tests/integration/$FILE
done
python -m pytest --cov=es2hfa --cov-report term-missing tests

