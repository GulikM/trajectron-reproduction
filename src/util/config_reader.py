import yaml
import os
import re

from src.util import DotDict

PATH_DEFAULT_CONFIG = "../config/_config.yml"
PATH_CONFIG = "../config/config.yml"
PATH_HYPERPARAMS = "../config/hyperparams.yml"
_DEFAULT_CONFIG = None
_HYPERPARAMS = None


def read_config(*args, **kwargs) -> DotDict:
    try:
        source_folder = (lambda a: a[:a.rindex("src") + 4])(os.getcwd())
    except ValueError:
        if "src" in os.listdir(os.getcwd()):
            source_folder = os.getcwd() + "/src"
        else:
            raise FileNotFoundError(f"Could not locate src folder from directory {os.getcwd()}")
    config_files = {
        file_name.rsplit(".", 1)[0]: os.path.join(item[0], file_name)
        for item in os.walk(source_folder)
        for file_name in item[2]
        if os.path.basename(item[0]) == "config" and file_name.endswith(".yml")
    }
    final_dict = DotDict()
    default_elements = dict()
    for name, file_name in config_files.items():
        if name.endswith("_default"):
            default_elements[name] = name[:name.find("_default")]
            continue
        with open(file_name, *args, **kwargs) as yaml_file:
            final_dict[name] = DotDict(yaml.load(yaml_file, Loader=yaml.FullLoader))

    for default_name, target_name in default_elements:
        with open(config_files[default_name], *args, **kwargs) as yaml_file:
            dict2 = DotDict(yaml.load(yaml_file, Loader=yaml.FullLoader))
            final_dict[target_name] = {**dict2, **final_dict[target_name]}
    return final_dict


class ConfigDict(DotDict):
    def __init__(self, *file_paths: str, **kwargs):
        result = {}
        for file_path in file_paths:
            with open(file_path) as f:
                config_element = yaml.load(f, Loader=yaml.FullLoader)
                result = {**result, **dict(config_element)}
        super().__init__(**{**result, **kwargs})
        pass
