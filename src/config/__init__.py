import inspect
import sys
from src.util import *
_parser = ConfigParser()
_CONFIG = _parser.read_config()


def get():
    return _CONFIG


def update(func):
    global _CONFIG, _parser

    def call_to_parser(*args, **kwargs):
        global _CONFIG, _parser
        result = func(*args, **kwargs)
        _CONFIG = _parser.read_config()
        return result
    return call_to_parser


@update
def with_aliases(names, others):
    return _parser.with_aliases(names, others)


@update
def with_alias(name, other):
    return _parser.with_alias(name, other)


@update
def omit(*names):
    return _parser.omit(*names)


def assign(dd: DotDict, name: str = None):
    def assign_decorator(func):
        dd[name or func.__name__] = func
        return func
    return assign_decorator


banned_items = inspect.getmembers(sys.modules[__name__], predicate=inspect.isfunction)


def __getattr__(name):
    for func_name, func in banned_items:
        if name == func_name:
            return func
    return _CONFIG[name]
