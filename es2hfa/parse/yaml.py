"""
Parsing the yaml input file
"""
import yaml


class YamlParser:
    """
    Parser for the input YAML text
    """

    @staticmethod
    def parse_str(string: str) -> dict:
        """
        Parse a string in the YAML format into the corresponding dictionary
        """
        return yaml.safe_load(string)

    @staticmethod
    def parse_file(input_file: str) -> dict:
        """
        Parse a YAML file into the corresponding dictionary
        """
        with open(input_file, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        return data_loaded
