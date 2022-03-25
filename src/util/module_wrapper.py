import inspect
import types
from typing import Union, TypeVar, Callable

Module = Union[types.ModuleType, type]
T = TypeVar('T')
AnyCount = Union[T, list[T]]


def nested_classes(source: AnyCount[Module], parents: AnyCount[type] = None) -> list[tuple[str, type]]:
    """
    Obtains a shallow list of all classes in the source or sources which are a child of the parents.
    :param source: One or more source elements
    :param parents: One or more valid children
    :return: list of ('ClassName', <class source.ClassName>)
    """
    if not isinstance(source, list):
        source = [source]

    array_results = [inspect.getmembers(child, predicate=_generate_predicate(parents)) for child in source]
    return sum(array_results, [])


def _generate_predicate(parents: AnyCount[Module]) -> Callable[[Module], bool]:
    if not parents:
        def _predicate(child: Module) -> bool:
            return inspect.isclass(child)
        return _predicate
    if not isinstance(parents, list):
        parents = [parents]

    def _predicate_subclasses(child: Module) -> bool:
        return inspect.isclass(child) and any(issubclass(child, parent) for parent in parents)
    return _predicate_subclasses


def generate(class_type: Module):
    def create_instance(*args, **kwargs):
        signature = inspect.signature(class_type.__init__)
        if len(args) == 0 or args[0] != class_type:
            args = tuple([class_type]) + args
        try:
            bound_arguments = signature.bind(*args, **kwargs)
        except TypeError as t:
            raise
        bound_arguments.apply_defaults()
        return class_type(*(bound_arguments.args[1:]), **bound_arguments.kwargs)    # we trim the parameter "self".
    return create_instance


class ModuleWrapper:
    def __init__(self, source: AnyCount[Module], parents: AnyCount[Module] = None):
        self.source = source if isinstance(source, list) else [source]
        self.parents = parents if isinstance(parents, list) or not parents else [parents]
        self.nested = nested_classes(source, parents)
        for name, cls in self.nested:
            setattr(ModuleWrapper, name, staticmethod(generate(cls)))
    pass
