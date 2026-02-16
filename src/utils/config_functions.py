# Import libraries
import yaml
from src.utils.logger_functions import console


def load_configs(path_config="../config/config.yaml"):
    # Parse config.yaml
    with open(path_config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            console.error(f"Failed to load config file. Received error: {e}")
        else:
            return config
