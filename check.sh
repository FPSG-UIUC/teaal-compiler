#!/bin/bash
# file name: check.sh

# Type Checking
pipenv run mypy es2hfa

# Auto-Formatting
pipenv run autopep8 -iraa es2hfa/
pipenv run autopep8 -iraa tests/

# Testing
for FILE in $(ls tests/integration | grep .hfa)
do
    perl -pi -e 'chomp if eof' tests/integration/$FILE
done
pipenv run python -m pytest --cov=es2hfa --cov-report term-missing tests

