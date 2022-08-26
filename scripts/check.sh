#!/bin/bash
# file name: check.sh

# Type Checking
mypy teaal

# Auto-Formatting
autopep8 -iraa teaal/
autopep8 -iraa tests/

# Testing
for FILE in $(ls tests/integration | grep .hfa)
do
    perl -pi -e 'chomp if eof' tests/integration/$FILE
done
python -m pytest --cov=teaal --cov-report term-missing tests

