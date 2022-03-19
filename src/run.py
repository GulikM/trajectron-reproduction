from src.util import config_reader


def run():
    pass


if __name__ == "__main__":
    conf = config_reader.get_config()
    hyperparams = config_reader.get_hyperparams()
    print(config_reader)
    print(hyperparams)
    run()
