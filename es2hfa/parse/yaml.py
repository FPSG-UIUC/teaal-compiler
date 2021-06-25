"""
Parsing the yaml input file
"""
# Known issue with ruamel and mypy (https://github.com/python/mypy/issues/7276)
from ruamel.yaml import YAML  # type: ignore


class YamlParser:
    """
    Parser for the input YAML text
    """

    @staticmethod
    def parse_str(string: str) -> dict:
        """
        Parse a string in the YAML format into the corresponding dictionary
        """
        yaml = YAML(typ='safe', pure=True)
        return yaml.load(string)

    @staticmethod
    def parse_file(input_file: str) -> dict:
        """
        Parse a YAML file into the corresponding dictionary
        """
        with open(input_file, 'r') as stream:
            yaml = YAML(typ='safe', pure=True)
            data_loaded = yaml.load(stream)
        return data_loaded
