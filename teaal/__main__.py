"""
MIT License

Copyright (c) 2021 University of Illinois

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os  # pragma: no cover
import sys  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
    # Configure Python path
    path = os.path.abspath(".")
    if path not in sys.path:
        sys.path.append(path)

    # Import the necessary classes
    from teaal.parse import *
    from teaal.trans.hifiber import HiFiber

    # Make sure we are given exactly one argument
    if len(sys.argv) != 2:
        print("Usage: pipenv run python teaal [input file]")

    # Translate
    else:
        einsum = Einsum.from_file(sys.argv[1])
        mapping = Mapping.from_file(sys.argv[1])
        arch = Architecture.from_file(sys.argv[1])
        bindings = Bindings.from_file(sys.argv[1])
        format_ = Format.from_file(sys.argv[1])
        hifiber = HiFiber(einsum, mapping, arch, bindings, format_)
        print(hifiber)
