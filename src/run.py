from pprint import pprint
import src.config as config


def run():
    pprint(config.get())


if __name__ == "__main__":
    run()
