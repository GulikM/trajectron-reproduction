import typing
from typing import Union

import ruamel.yaml as yaml
import os
from pathlib import Path
from .dot_dict import DotDict
from .mapping_dict import MappingDict
from .config_loader import ConfigLoader


class ConfigParser:
    """
    Class for parsing config data using a ConfigLoader.
    """
    def __init__(self,
                 filetypes: Union[str, list[str]] = None,
                 top_folder: str = "src",
                 target_folders: Union[list[str], str] = "config",
                 curr_dir: Union[str, Path] = Path(os.getcwd()).absolute(),
                 loader: typing.Callable = yaml.safe_load,
                 aliases: dict = None,
                 omit_files: Union[str, list[str]] = "",
                 **kwargs):

        self.filetypes = filetypes if isinstance(filetypes, list) else (
            [filetypes] if filetypes else [".yml", ".yaml", ".json"]
        )
        self.top_folder = top_folder
        self.target_folders = target_folders if isinstance(target_folders, list) else [target_folders]
        self.curr_dir = curr_dir if isinstance(curr_dir, Path) else Path(curr_dir).absolute()
        self.loader = loader
        self.aliases = MappingDict(aliases or dict())
        self.reader_config = kwargs
        self.omitted_files = omit_files if isinstance(omit_files, list) else [omit_files]

        if self.curr_dir.is_file():
            self.curr_dir = self.curr_dir.parents[0]
        if not self.curr_dir.is_dir():
            raise ValueError("Current path specified is invalid!")
        pass

    def with_aliases(self, names: typing.Iterable[str], others: typing.Iterable[str]):
        """
        Define aliases for config files. Requires one-to-one correspondence between lists.
        :param names: config file names (without file extensions (e.g. ".yml"))
        :param others: rename targets
        :return: self
        """
        for i_name, i_other in zip(names, others):
            self.with_alias(i_name, i_other)
        return self

    def with_alias(self, name: str, other: str):
        """
        Define an alias for a given file name.
        :param name: config file name without file extension
        :param other: rename target
        :return: self
        """
        if name in self.omitted_files:
            self.omitted_files.remove(name)
        if (name == other or other is None) and name in self.aliases:
            del self.aliases[name]
            return
        if not (name and other):
            raise ValueError("Empty strings encountered")
        self.aliases[name] = other
        return self

    def omit(self, *args: str):
        """
        Omits a file name from the result.
        :param args:
        :return:
        """
        self.omitted_files += list(args)
        return self

    def find_top_dir(self, folder: str) -> str:
        """
        Obtain full path of top directory. Raises ValueError if top directory cannot be found by crawling up.
        :param folder: Name of folder
        :return: full path of folder
        """
        parents = self.curr_dir.parents
        try:
            top = next((child[0] for p in parents for child in os.walk(p) if Path(child[0]).stem == folder))
        except StopIteration:
            raise ValueError(f'Could not locate "{folder}" from {self.curr_dir}!')
        return top

    def read_config(self, *args, **kwargs) -> DotDict:
        """
        Function for reading YAML config files as presented in src.config
        :param args, kwargs: additional parameters for reader opening
        :return: DotDict representation of configuration
        """
        top = self.top_folder
        if not os.path.exists(top):
            top = self.find_top_dir(top)

        file_names = [os.path.join(folder[0], filename)
                      for folder in os.walk(top) if os.path.basename(folder[0]) in self.target_folders
                      for filename in folder[2]
                      if (lambda fn: fn.suffix in self.filetypes and fn.stem not in self.omitted_files)(Path(filename))]

        file_readers = [ConfigLoader(data=file_name, **self.reader_config)
                        for file_name in file_names]

        file_data = [(name, value if isinstance(value, dict) else self.loader(value))
                     for name, value in map(lambda cl: cl.load(), file_readers)]
        return DotDict({
            self.aliases[name]: value for name, value in file_data
        })
