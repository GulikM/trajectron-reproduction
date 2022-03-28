import inspect
from typing import Union


class Validated:
    """
    Base for any class that wishes to plainly implement validation. Any methods within these
    classes which start with "validate" and have no required arguments shall be called.

    Note that these functions are only called. Use exceptions or warnings within these class
    methods to assert an effect upon the user.
    """
    def _call_all(self, start: Union[list[str], str]):
        """
        Method for calling all functions of class, barring those inherited from Validated class.
        :param start: string or list containing the prefixes required for calling.
        :result: calls all methods.
        """
        if start and not isinstance(start, list):
            start = [start]

        def _predicate(member):
            if not inspect.ismethod(member):
                return False    # assert that member is a method
            if member.__name__ in Validated.__dict__:
                return False    # assert that member is not a method of class Validated
            if start and not any(member.__name__.startswith(s_string) for s_string in start):
                return False    # assert that member adheres to namespace provided, if any
            if any(p.default is inspect.Parameter.empty for n, p in inspect.signature(member).parameters.items()):
                return False    # assert that member has either 1) no params; or 2) a default value for each param
            return True
        [func() for _, func in inspect.getmembers(self, predicate=_predicate)]

    def validate(self):
        """
        Validates all methods
        """
        self._call_all('validate')
