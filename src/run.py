from pprint import pprint
import src.config as config


@config.assign(config.my_methods.example_methods, "example_function")
def multiply_by_seven(k: int):
    return 7 * k


def run():
    run_mode = config.operation.mode

    if not isinstance(run_mode, str):
        print("No run mode assigned. Defaulting to train.")
        config.operation.mode = "train"
    else:
        print(f"In mode {config.operation.mode}...")

    example_output = config.my_methods.example_methods.example_function(6)
    print(example_output)

    pprint(config.get())
    print(config.defaults.H)


if __name__ == "__main__":
    run()
