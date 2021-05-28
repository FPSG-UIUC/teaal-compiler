# Einsum to HFA Compiler
![Build Status](https://github.com/FPSG-UIUC/hfa-compiler/actions/workflows/test.yml/badge.svg)


## Installation

First, make sure you have [pipenv](https://pipenv.pypa.io/en/latest/).

To set up the virtual environment, run
```
pipenv install
```

You can then run any commands in the virtual environment with
```
pipenv run [cmd]
```
or
```
pipenv shell
[cmd]
```

## All Checks

All checks can be run with the command
```
./check.sh
```
The details for how to run specific checks can be found below.

## Type Checking

To type check, run
```
pipenv run mypy es2hfa
```

## Linting

To lint, run
```
pipenv run autopep8 -iraa es2hfa/
pipenv run autopep8 -iraa tests/
```

Note that this uses the most aggressive form of linting available with
autopep8. We can always reduce the amount of linting by using one `-a` (less
agressive code changes) or none (whitespace changes only).


## Tests

To run tests, run
```
pipenv run python -m pytest tests
```
and to get the test coverage statistics, run
```
pipenv run python -m pytest --cov=es2hfa --cov-report term-missing tests
```
