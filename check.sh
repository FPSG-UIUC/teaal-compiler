#!/bin/bash
# file name: check.sh

pipenv run mypy es2hfa
pipenv run autopep8 -iraa es2hfa/
pipenv run autopep8 -iraa tests/
pipenv run python -m pytest --cov=es2hfa --cov-report term-missing tests

