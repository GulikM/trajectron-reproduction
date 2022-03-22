import random
import re
import typing
from collections import defaultdict
from functools import partial

from src.util.list_available_functions import *
from src.util.aggregation import *


class DotDict(defaultdict):
    """dot.notation access to dictionary attributes"""
    _banned_keywords = None
    _regex_omit = r"[^a-zA-Z]*"
    _regex_wrap = r"{0}{{0}}{0}".format(_regex_omit)

    def __init__(self, *args, **kwargs):
        n_args = (DotDict._recursive_conversion(arg, True) for arg in args)
        n_kwargs = DotDict._recursive_conversion(kwargs, True)
        super().__init__(DotDict, *n_args, **n_kwargs)
        pass

    @classmethod
    def _set_banned_keywords(cls):
        if cls._banned_keywords:
            return
        cls._banned_keywords = set(sum(map(
            lambda member:
                [member, re.sub(pattern=cls._regex_omit, repl="", string=member)],
            inspect.getmembers(cls)
        )))
        # ... That's a lot - let's unpack:
        # map( function(tuple)->list, list[tuple]) -> list[list]    >> convert inspect.getmembers into argument names
        # sum(list[list]) -> list                                   >> use "sum" to concatenate list of lists
        # set(list) -> set                                          >> convert to set to attain uniqueness
        pass


    def assert_valid_keywords(self, *args, **kwargs):
        pass

    @classmethod
    def assert_valid_keyword(cls, elem) -> bool:
        pass


    @staticmethod
    def _recursive_conversion(element, omit_dict=False):
        if isinstance(element, dict):
            recursive_converted = {k: DotDict._recursive_conversion(v) for k, v in element.items()}
            return DotDict(recursive_converted) if not omit_dict else recursive_converted
        elif isinstance(element, typing.Iterable) and not isinstance(element, str):
            return type(element)(DotDict._recursive_conversion(sub_element) for sub_element in element)
        else:
            return element

    __getattr__ = defaultdict.__getitem__
    __setattr__ = defaultdict.__setitem__
    __delattr__ = defaultdict.__delitem__
