from enum import Enum
import os
from pathlib import Path
import inspect
from typing import Union


class ConfigLoader:
    def __new__(cls, data: Union[str, dict], **kwargs):
        if isinstance(data, dict):
            return ConfigDictLoader.__new__(ConfigDictLoader)
        elif not isinstance(data, str):
            raise TypeError(f"Type '{type(data)}' is not supported. Supported types: [str, dict].")
        if os.path.exists(data):
            return ConfigFileLoader.__new__(ConfigFileLoader)
        return ConfigTextLoader.__new__(ConfigTextLoader)

    def load(self) -> tuple[str, Union[str, dict]]:
        """
        Loading of relevant data.
        :return: a tuple containing first the name of this config element, followed by the relevant data.
        """
        raise NotImplementedError("Method must be overridden!")


class ConfigLoaderSubclass(ConfigLoader):
    __new__ = object.__new__
    load = dict


class ConfigFileLoader(ConfigLoaderSubclass):
    def __init__(self, data, *args, **kwargs):
        self.base_path = data
        self.signature = inspect.signature(open).bind("", *args, **kwargs)

    def load(self) -> tuple[str, str]:
        with open(self.base_path, *self.signature.args[1:], **self.signature.kwargs) as file:
            return Path(self.base_path).stem, ''.join(file.readlines())


class ConfigTextLoader(ConfigLoaderSubclass):
    def __init__(self, data: str, *args, **kwargs):
        self.config_data = data
        self.name = kwargs.get("name") or str(id(self))

    def load(self) -> tuple[str, str]:
        return self.name, self.config_data


class ConfigDictLoader(ConfigLoaderSubclass):
    def __init__(self, data: dict, *args, **kwargs):
        self.config_data = data
        if "name" in kwargs:
            self.name = kwargs.get("name")
        elif len(self.config_data.items()) == 1:
            self.name = next(k for k in self.config_data.keys())
            self.config_data = self.config_data[self.name]
            if not isinstance(self.config_data, dict):
                self.config_data = {"values": self.config_data}
        else:
            self.name = str(id(self))

    def load(self) -> tuple[str, dict]:
        return self.name, self.config_data
