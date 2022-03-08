import yaml

PATH_DEFAULT_CONFIG = "../config/default.yml"
PATH_CONFIG = "../config/config.yml"


##
def read_config(default: str = PATH_DEFAULT_CONFIG, config: str = PATH_CONFIG) -> dict:
    """
    Reading YAML files and loading their config, keeping default values where no "new" ones are provided.
    :param default: the default configuration
    :param config: the specified configuration
    :return: dictionary (dict) of configuration
    """
    with open(default) as f_default, open(config) as f_config:
        d_default, d_config = yaml.load(f_default, Loader=yaml.FullLoader), yaml.load(f_config, Loader=yaml.FullLoader)
        result = {**dict(d_default), **dict(d_config)}
    return result


_config = read_config()


def get() -> dict:
    """
    Method to obtain a (copy of) configuration.
    :return: Configuration dictionary.
    """
    return _config.copy()
