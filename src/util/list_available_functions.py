import inspect
from typing import Union, Any, Callable, Type


def _generate_predicate(parents: Union[Type, list[Type]]) -> Callable[[Type], bool]:
    if not parents:
        def _predicate(child: Type) -> bool:
            return inspect.isclass(child)
        return _predicate
    if not isinstance(parents, list):
        parents = [parents]

    def _predicate_subclasses(child: Type) -> bool:
        return inspect.isclass(child) and any(issubclass(child, parent) for parent in parents)
    return _predicate_subclasses


def nested_classes(children: Union[Type, list[Type]], parents: Union[Type, list[Type]] = None) -> list[tuple[str, Any]]:
    if not isinstance(children, list):
        children = [children]

    array_results = [inspect.getmembers(child, predicate=_generate_predicate(parents)) for child in children]
    return sum(array_results, [])


def create_instance(class_type: type, *args, **kwargs):
    signature = inspect.signature(class_type.__init__)
    if len(args) == 0 or args[0] != class_type:
        args = tuple([class_type]) + args
    try:
        bound_arguments = signature.bind(*args, **kwargs)
    except TypeError as t:
        raise
    bound_arguments.apply_defaults()
    return class_type(*(bound_arguments.args[1:]), **bound_arguments.kwargs)    # we trim the parameter "self".
