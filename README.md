# Einsum to HFA Compiler

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
