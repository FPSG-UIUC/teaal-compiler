# Einsum to HiFiber Compiler
![Build Status](https://github.com/FPSG-UIUC/teaal-compiler/actions/workflows/test.yml/badge.svg)


## Installation

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
pip install -e git+ssh://git@github.com/FPSG-UIUC/teaal-compiler.git#egg=teaal
```

## All Checks

All checks can be run with the command
```
./scripts/check.sh
```
The details for how to run specific checks can be found below.

## Type Checking

To type check, run
```
mypy teaal
```

In order for the YAML parser to typecheck, you may need to add an empty
`__init__.pyi` in the directory
`env/lib/python3.<#>/site-packages/ruamel/yaml/`.

## Linting

To lint, run
```
autopep8 -iraa teaal/
autopep8 -iraa tests/
```

Note that this uses the most aggressive form of linting available with
autopep8. We can always reduce the amount of linting by using one `-a` (less
agressive code changes) or none (whitespace changes only).


## Tests

To run tests, run
```
python -m pytest tests
```
and to get the test coverage statistics, run
```
python -m pytest --cov=teaal --cov-report term-missing tests
```
