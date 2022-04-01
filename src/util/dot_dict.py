import re
import sys
import types
import typing
from typing import Any
import keyword
from collections import defaultdict
from src.util.module import *


class DotDict(defaultdict):
    """
    Class for creating a dot-accessible dictionary ("DotDict").
    @author Dirk Remmelzwaal
    """
    _reserved_members = None
    _regex_omit = r"[^a-zA-Z]*"
    _regex_wrap = r"{0}{{0}}{0}".format(_regex_omit)

    __delattr__ = defaultdict.__delitem__
    __str__ = dict.__str__

    def __init__(self, *args, _recursive_call: bool = False, trace: str = None, **kwargs):
        if not _recursive_call:
            [DotDict.assert_validity(arg, trace=trace) for arg in args]
            DotDict.assert_validity(kwargs, trace=trace)
        super().__init__(DotDict, *args, **kwargs)
        pass

    def __repr__(self):
        """
        Generate string representation
        :return: String representation for reconstructing this object
        """
        return "DotDict({0})".format(
            dict.__repr__(self)
        )

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set self[key] to value
        :param key:
        :param value:
        """
        assert DotDict.assert_validity(key)
        self.__setitem__(key, value if not isinstance(value, dict) or isinstance(value, DotDict) else DotDict(value))
        pass

    def __getattr__(self, key: str) -> Any:
        """
        Overriding the attribute-getattr magic method for dict-like behaviour
        :param key: key of dict element
        :return: dict element at key; defaults to a new inplace DotDict
        """
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> Any:
        """
        Overriding the attribute-getitem magic method for nested type assignment
        :param key: key of dict element
        :return: dict element at key; defaults to a new inplace Dotdict
        """
        result = defaultdict.__getitem__(self, key)
        if result is None:
            print("HUHHHHH+")
        if isinstance(result, dict) and not isinstance(result, DotDict):
            self.__setattr__(key, DotDict(result))
        return defaultdict.__getitem__(self, key)

    def items(self):
        result = super().items()
        return [(k, v if not isinstance(v, dict) else DotDict(v)) for k, v in result]

    def values(self):
        print("I was called!")
        result = super().values()
        return [v if not isinstance(v, dict) else DotDict(v) for v in result]

    @classmethod
    def is_reserved_member(cls, word: str) -> bool:
        """
        Asserts if a given string is reserved as an attribute of self
        :param word: string to assert
        :return: True if word is reserved else False
        """
        return word in cls.get_reserved_members()

    @classmethod
    def get_reserved_members(cls) -> list[str]:
        """
        Obtain a list of all attribute names of self
        :return: list of all attributes
        """
        cls._set_reserved_members()
        return cls._reserved_members

    @classmethod
    def _set_reserved_members(cls) -> None:
        """
        Internal method for initiating list of all attributes of self
        """
        if cls._reserved_members:
            return
        cls._reserved_members = set(sum(map(
            lambda member:
            [member[0], re.sub(pattern=cls._regex_omit, repl="", string=member[0])],
            inspect.getmembers(cls)
        ), []))
        # ... That's a lot - let's unpack:
        # map( function(tuple)->list, list[tuple]) -> list[list]    >> convert inspect.getmembers into argument names
        # sum(list[list]) -> list                                   >> use "sum" to concatenate list of lists
        # set(list) -> set                                          >> convert to set to attain uniqueness
        pass

    @classmethod
    def assert_validity(
            cls, elem, trace: Union[str, list[Union[int, str]]] = None, eval_str: bool = True
    ) -> bool:
        """
        Asserts if elem does not violate restrictions placed on keys in DotDicts.
        :param elem: element to assert
        :param trace: debugging tool:
        :param eval_str: evaluate strings; used to omit non-key strings from check.
        :return:
        """
        if not isinstance(trace, list):
            trace = [trace]

        def _assert_validity_str(elem_str: str, n_trace: list[Union[int, str]]) -> bool:
            if exception_reason := cls.violated_keyword_rules(elem_str):
                raise IllegalKeywordWarning(path=n_trace, illegal_keyword=elem_str, reason=exception_reason)
            return True

        def _assert_validity_dict(elem_dict: dict, n_trace: list[Union[int, str]]) -> bool:
            if not isinstance(elem_dict, dict):
                return True
            return all(_assert_validity_str(k, n_trace) and (
                isinstance(v, str) or cls.assert_validity(v, n_trace + [k])
            ) for k, v in elem_dict.items())

        def _assert_validity_list(elem_list: typing.Iterable, n_trace: list[Union[int, str]]) -> bool:
            return all(cls.assert_validity(nested, n_trace + [index], False) for index, nested in enumerate(elem_list))

        try:
            if isinstance(elem, str):
                return not eval_str or _assert_validity_str(elem, trace)
            if isinstance(elem, dict):
                return _assert_validity_dict(elem, trace)
            if isinstance(elem, typing.Iterable):
                return _assert_validity_list(elem, trace)

        except IllegalKeywordWarning as e:
            stack = inspect.stack()
            fr = next(fr.frame for fr in stack if fr.filename != stack[0].filename) or stack[0].frame
            tb = None
            tb = types.TracebackType(tb, fr, fr.f_lasti, fr.f_lineno)
            raise e.with_traceback(tb)
        return True

    @classmethod
    def violated_keyword_rules(cls, word: str) -> str:
        return '; '.join(filter(None, [
            'word "{keyword}" is a reserved class member' if cls.is_reserved_member(word) else None,
            'word "{keyword}" is reserved by Python' if keyword.iskeyword(word) else None,
            'word "{keyword}" contains illegal characters' if re.search(r"\W", word) else None
        ]))


class IllegalKeywordWarning(ResourceWarning):
    message = 'Illegal keyword "{{keyword}}" found{{path}}{reason}'

    def __init__(self, path=None, illegal_keyword=None, reason=None):
        if isinstance(path, typing.Iterable):
            path = list(filter(None, path))
        super().__init__(
            self.message.format(
                reason="" if not reason else '\nReason: {0}'.format(reason)
            ).format(
                keyword=illegal_keyword or "",
                path="" if not path else ' at location {0}'.format(".".join(map(lambda elem: str(elem), path))),
            )
        )
