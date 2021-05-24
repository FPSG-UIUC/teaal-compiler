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

## Tests

To run tests, run
```
pipenv run python -m pytest tests
```
and to get the test coverage statistics, run
```
pipenv run python -m pytest --cov=es2hfa tests
```
