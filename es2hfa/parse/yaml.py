"""
Parsing the yaml input file
"""
import yaml
import io


class YamlParser:
    """
    Parser for the input yaml file
    """

    @staticmethod
    def parse(input_file: str) -> dict:
        # Read YAML file
        with open(input_file, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        return data_loaded
