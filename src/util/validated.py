import inspect


class Validated:
    def __call_all(self, start: str):
        """
        Method for calling all methods starting with "validate_"
        """
        def _predicate(member):
            if not inspect.ismethod(member):
                return False
            if member.__name__ in Validated.__dict__:
                return False
            if any(p.default is inspect.Parameter.empty for n, p in inspect.signature(member).parameters.items()):
                return False
            if start and not member.__name__.startswith(start):
                return False
            return True
        [func() for _, func in inspect.getmembers(self, predicate=_predicate)]

    def validate(self):
        self.__call_all('validate')