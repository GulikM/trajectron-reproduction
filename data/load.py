import inspect
import pandas as pd

from pathlib import Path
from typing import Union

class Dataset(object):
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = path # call setter
        self._data = None
    
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

    def load(self, *args, **kwargs) -> None:
        # keep track of passed parameters to enable reloading of the dataset
        self._load_parameters = {
            'args': args,
            'kwargs': kwargs
        }

    def reload(self):
        params = getattr(self, '_load_parameters', None)
        if params is None:
            raise NotImplementedError
        self.load(*params.args, **params.kwargs)

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
        return self._data.columns.tolist()

    @property
    def index(self):
        return self._data.index.name

    @index.setter
    def index(self, col: str) -> None:
        if not col in (header := self.header):
            raise ValueError(f'{col} not in {header}')
        self._data.set_index([col])

    def validate_header(self) -> bool:
        if self.__class__.required_columns is not None:
            if not self.__class__.required_columns in self.header:
                raise NotImplementedError