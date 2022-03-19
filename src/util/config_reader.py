import yaml
import os

PATH_DEFAULT_CONFIG = "../config/default.yml"
PATH_CONFIG = "../config/config.yml"
PATH_HYPERPARAMS = "../config/hyperparams.yml"
_DEFAULT_CONFIG = None
_HYPERPARAMS = None

_SOURCE_FOLDER = (lambda a: a[:a.rindex("src") + 4])(os.getcwd())
_CONFIG_FILES = [
    os.path.join(item[0], file_name)
    for item in os.walk(_SOURCE_FOLDER)
    for file_name in item[2]
    if os.path.basename(item[0]) == "config" and file_name.endswith(".yml")
]
_CONFIG_FILE_NAMES = [os.path.basename(file_path).rsplit('.', 1)[0] for file_path in _CONFIG_FILES]


class ConfigReader:
    def __init__(self, base: str = PATH_DEFAULT_CONFIG, *additional: str):
        self.files = [base] + list(additional)
        self.config = None
        pass

    def read_config(self) -> None:
        """
        Reading YAML files and loading their config, keeping default values where no "new" ones are provided.
        :param default: the default configuration
        :param config: the specified configuration
        :return: dictionary (dict) of configuration
        """
        result = {}
        for file in self.files:
            with open(file) as f:
                config_element = yaml.load(f, Loader=yaml.FullLoader)
                result = {**result, **dict(config_element)}
        self.config = result
        pass

    def get(self) -> dict:
        """
        Method to obtain a (copy of) configuration.
        :return: Configuration dictionary.
        """
        if not self.config:
            self.read_config()
        return self.config.copy()


def get_config() -> dict:
    global _DEFAULT_CONFIG
    if not _DEFAULT_CONFIG:
        _DEFAULT_CONFIG = ConfigReader()
    return _DEFAULT_CONFIG.get()


def get_hyperparams() -> dict:
    global _HYPERPARAMS
    if not _HYPERPARAMS:
        _HYPERPARAMS = ConfigReader(base=PATH_HYPERPARAMS)
    return _HYPERPARAMS.get()
