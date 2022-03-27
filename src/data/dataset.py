import inspect
import pandas as pd

from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class Dataset(object):
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = path # call setter
        self._data = None
        self.params = None # TODO: rename

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

    def load(self, overwrite_params: bool = False, **kwargs) -> None:
        # keep track of passed parameters to enable reloading of the dataset
        if not overwrite_params:
            self.params = {**(self.params or dict()), **kwargs}
        else:
            self.params = kwargs

    def unload(self, *keys: str) -> None:
        # unload keys by name
        for key in keys:
            if self.params and key in self.params:
                del self.params[key]

    def __call_all(self, startswith: str):
        members = inspect.getmembers(self, predicate=inspect.ismethod) # class methods
        for name, value in members:
            if name.startswith(startswith) and callable(value):
                value()

    def validate(self):
        self.__call_all('validate_')


class CSVDataset(Dataset):  
    required_columns: Optional[List[str]] = None

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path)

    def load(self, overwrite_params: bool = False, **kwargs) -> None: 
        super().load(overwrite_params, **kwargs) # keep track of passed parameters to enable reloading of the dataset 
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

    def loader(self):
        raise NotImplementedError # TODO: add generator functionality so that the whole dataset does not necessarily have to be loaded into memory at once

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

    def validate_header(self) -> None:
        if not (required := self.__class__.required_columns):
            return
        for col in required:
            if not col in self.header:
                raise NotImplementedError # TODO: add appropriate exception

    def filter(self, row_filters: Optional[Dict[str, Any]] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:        
        '''
        

        Args:
            row_filters: 
            columns:

        Returns:
            A subset of the dataset
        '''
        if not columns:
            columns = self.header  # select all valid columns
        
        if not row_filters:
            return self.data[columns]
    
        masks = []
        # create boolean mask per row filter
        for key, value in row_filters.items():
            if not key in columns:
                raise KeyError(f'{key} not in {columns}')
            else:
                masks.append(self.data[key].isin(value))
        
        rows = [all(tup) for tup in zip(*masks)] # combine masks 
        return self.data[rows, columns]





    # def pop(self, item: str, transpose: bool = False, copy: bool = False) -> pd.Series:
    #     '''
    #     Return item and drop from frame. Raise KeyError if not found.
        
    #     Args:
    #         item: Label of the row/column to be popped
    #         copy: Whether to copy the data after transposing, even for DataFrames 
    #               with a single dtype. Note that a copy is always required for 
    #               mixed dtype DataFrames, or for DataFrames with any extension types.
    #     Returns:
    #         ....
    #     '''
    #     if transpose:
    #         df = self.data.transpose(copy=copy) # TODO: check if better to assign _data property directly to df instead of calling data getter
    #     else:
    #         df = self.data
    #     return df.pop(item)
