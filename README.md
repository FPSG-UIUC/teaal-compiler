# Einsum to HFA Compiler
![Build Status](https://github.com/FPSG-UIUC/hfa-compiler/actions/workflows/test.yml/badge.svg)


## Installation

### Pipenv

Pipenv automatically manages the virtual environment and dependencies for you.
It is the easiest way to get started.

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

### Manual Virtual Environment

First, make sure you have [venv](https://docs.python.org/3/library/venv.html).

To create the virtual environment,
```
python3 -m venv env
```

Then to enter the virtual environment, run
```
source env/bin/activate
```

Install all packages including the compiler with
```
pip install -e .
```

or remotely via SSH with
```
pip install -e git+ssh://git@github.com/FPSG-UIUC/teaal-compiler.git#egg=es2hfa
```

If installing this way, remember to remove `pipenv run` from the beginning of
each of the commands.

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

In order for the YAML parser to typecheck, you may need to add an empty
`__init__.pyi` in the directory
`~/.local/share/virtualenvs/teaal-compiler-<hash>/lib/python3.<#>/site-packages/ruamel/yaml/`.

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
