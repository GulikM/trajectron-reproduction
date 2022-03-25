import inspect
import pandas as pd

from pathlib import Path
from typing import Union


class Validated(object):
    def __call_all(self, start: str):
        def _predicate(member):
            return inspect.isfunction(member) and (not start or member[0].startswith(start))
        [func() for _, func in inspect.getmembers(self, predicate=_predicate)]

    def validate(self):
        self.__call_all('validate_')


class Foo(Validated):
    def validate_name(self):
        print(f"HEY I'm {type(self)}")

f = Foo()
f.validate()



class Dataset(object, Validated):
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = path # call setter
        self._data = None
        self._load_args = self._load_kwargs = None
        self.params = None
    
    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Union[str, Path]):
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise NotImplementedError  # TODO: raise appropriate exception
        elif not path.is_file():
            raise FileNotFoundError(f"No such file: '{path}'")
        self._path = path
    
    @property
    def name(self):
        return self._path.stem

    @property
    def data(self):
        return self._data

    def load(self, overwrite_params=False, **kwargs) -> None:
        # keep track of passed parameters to enable reloading of the dataset
        if not overwrite_params:
            self.params = {**(self.load_args or dict()), **kwargs}
        else:
            self.params = kwargs

    def unload(self, *keys: str):
        # unload keys by name
        for key in keys:
            if self.params and key in self.params:
                del self.params[key]

    def __call_all(self, startswith: str):
        members = inspect.getmembers(self, predicate=inspect.ismethod)
        for name, value in members:
            if name.startswith(startswith) and callable(value):
                value()

    def validate(self):
        self.__call_all('validate_')


class CSVDataset(Dataset):  
    required_columns = None

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path)

    def load(self, **kwargs) -> None:
        super().load(**kwargs) # keep track of passed parameters to enable reloading of the dataset 
        try:
            inspect.signature(pd.read_csv).bind_partial(**kwargs)
        except Exception as e:
            raise e
        chunksize = kwargs.pop('chunksize', None)
        if chunksize is None:
            self._data = pd.read_csv(self._path, **kwargs)
        else:
            chunks = []
            for chunk in pd.read_csv(self._path, **kwargs, chunksize=chunksize):
                chunks.append(chunk)
            self._data = pd.concat(chunks)

    @property
    def header(self):
        return self._data.columns.tolist() # columns

    @property
    def index(self):
        return self._data.index.name # column(s) used as the row labels of the dataframe

    @index.setter
    def index(self, col: str) -> None:
        if not col in (header := self.header):
            raise ValueError(f'{col} not in {header}')
        self._data.set_index([col])

    def validate_header(self) -> bool:
        if self.__class__.required_columns is not None:
            if not self.__class__.required_columns in self.header:
                raise NotImplementedError