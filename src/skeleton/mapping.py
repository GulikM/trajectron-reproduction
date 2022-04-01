from collections import defaultdict
from typing import Union

from src.util import ModuleWrapper, obtain_module
import src.config as config
import torch as tc
import regex as re


class ModuleMapping:
    def __init__(self):
        self.params = dict()
        self.modules = dict()
        self.inputs = dict()
        self.input_mapping = dict()
        self.outputs = dict()
        self.activation_wrapper = ModuleWrapper(tc.nn.modules.activation, tc.nn.Module)
        for name, elem in config.network.components.items():

            self.modules[name] = obtain_module(elem.class_name)(**elem.params)

            if not elem.output:
                raise ValueError(f"Component {name} does not have a specified output!")
            if elem.output in self.param_dependency:
                raise KeyError(f"Output {elem.output} was already assigned!")

            self.outputs[name] = elem.output

            if isinstance(elem.inputs, str):
                self.input_mapping[name] = {elem.inputs: None}
            elif isinstance(elem.inputs, list):
                self.input_mapping[name] = {input_value: None for input_value in elem.inputs}
            else:
                self.input_mapping[name] = elem.inputs

            self.inputs[name] = list(self.input_mapping[name].keys())

            for attr in self.input_mapping:
                self.__setattr__(attr, None)

    def __getattribute__(self, item):
        for resource_name in ["modules", "params"]:
            if item in super().__getattribute__(resource_name):
                return super().__getattribute__(resource_name)[item]
        return super().__getattribute__(item)

    def process_str(self, formula: str):
        split_str = re.split('\b', formula)
        split_str = re.sub(' *', '', split_str)
        split_str = [[elem] if not re.match('^[^a-zA-Z]*$', elem) else [i for i in elem] for elem in split_str]
        split_str = [k for elem in split_str for k in elem]
        split_str = [config.defaults[elem] for elem in split_str if elem in self.params]

    def activate(self, tensor: tc.Tensor, activation: Union[str, dict], **kwargs):
        if isinstance(activation, dict):
            return self.activate(tensor, activation["name"] or "sigmoid", **(activation["params"] or dict()))
        if activation.lower() not in self.activation_wrapper:
            raise ValueError(f"Activation function {activation} is not known.")
        return self.activation_wrapper[activation.lower()](**kwargs)(tensor)

    def forward(self, output: Union[str, list[str]] = None, **inputs: tc.Tensor):
        if not output:
            output = list(self.params.keys())

        updated_params = list(inputs.keys())
        for k, v in inputs.items():
            if k in self.params:
                self.params[k] = v

        staged_modules = list(self.modules.keys())

        stuck = False
        while len(staged_modules) > 0 and not stuck:
            stuck = True
            completed_mods = list()
            for mod_name in staged_modules:
                if all(param in updated_params for param in self.param_dependency[mod_name]):
                    stuck = False
                    args = [self.params[k] for k, v in self.input_mapping.items() if v is None]
                    kwargs = dict()
                    for k, v in self.input_mapping.items():
                        if v:
                            if isinstance(v, str):
                                kwargs[v] = self.params[k]
                            elif isinstance(v, dict):
                                alias = v["alias"] or v["key"] or v["name"]
                                activation = v["activation"]
                                act_kwargs = v["kwargs"] or v["params"] or dict()
                                kwargs[alias] = self.activate(self.params[k], activation, **act_kwargs)
                    out = self.outputs[mod_name]
                    if isinstance(out, str):
                        mod_output = out
                        mod_act = None
                        mod_act_kwargs = None
                    else:
                        mod_output = out["param"] or out["name"]
                        mod_act = out["activation"]
                        mod_act_kwargs = out["kwargs"] or out["params"] or dict()
                    self.params[mod_output] = self.modules[mod_name](*args, **kwargs)
                    if mod_act:
                        self.params[mod_output] = self.activate(self.params[mod_output], mod_act, **mod_act_kwargs)

                    updated_params.append(mod_output)
                    completed_mods.append(mod_name)

            for completed_mod in completed_mods:
                staged_modules.remove(completed_mod)


mm = ModuleMapping()

for key, value in mm.modules.items():
    print(f"{key}: {value}")
