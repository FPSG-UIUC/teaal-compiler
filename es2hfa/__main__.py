import os
import sys

if __name__ == "__main__":
    # Configure Python path
    path = os.path.abspath(".")
    if path not in sys.path:
        sys.path.append(path)

    # Import the necessary classes
    from es2hfa.parse.input import Input
    from es2hfa.trans.translate import Translator

    # Make sure we are given exactly one argument
    if len(sys.argv) != 2:
        print("Usage: pipenv run python es2hfa [input file]")

    # Translate
    else:
        input_ = Input(sys.argv[1])
        hfa = Translator.translate(input_)
        print(hfa.gen(0))
