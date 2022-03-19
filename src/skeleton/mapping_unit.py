import inspect

import torch as tc
import importlib
import src.util
from src.components.network_component import NetworkComponent
from src.util import nested_classes


class MappingUnit(tc.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        components = {k: v for k, v in kwargs.items() if isinstance(v, NetworkComponent)}
        for alias, network in components:
            self.__dict__[alias] = network


    def _register_activation_functions(self, **kwargs):
        """
        Loads all activation functions from the class
        :return:
        """
        relevant_classes = nested_classes(tc.nn.modules.activation)
        module = importlib.import_module(tc.nn.modules.activation)
