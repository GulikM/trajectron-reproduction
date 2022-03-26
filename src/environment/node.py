from typing import Optional

import pandas as pd

from .nodetype import NodeType


class Node(object):
    # instances = set()

    def __init__(self, type: Optional[NodeType], id: int, data: Optional[pd.DataFrame], is_robot: bool = False) -> None:
        self.type = type
        self.id = id
        self._data = data
        self.is_robot = is_robot
        # keep track of all node instances
        # Node.instances.append(self)

    @property
    def timestamps(self):
        return self._data['t'].unique()

    # @classmethod
    # def get(cls, key: str, value):
    #     for inst in cls.instances:
    #         if getattr(inst, key) == value:
    #             return inst

    # def get_neighbors_at_timestamp(self, timestamp: int):





    #     return [node for node in self.__class__.instances if node.id != self.id]


    # def time_window(self, upper_bound: int, lower_bound: int, cols: Optional[List[str]]):
    #     '''
    #     Method for 
        
    #     :param upper_bound: (inclusive)        
    #     :param lower_bound: (inclusive)
    #     :param cols:       
    #     :return:  
    #     '''
    #     if cols is None:
    #         data = self._data
    #     else:
    #         data = self._data[cols]
    #     mask1 = data['t'] >= lower_bound
    #     mask2 = data['t'] <= upper_bound
    #     mask = mask1 & mask2
    #     return data.loc[mask].to_numpy()

    # # TODO: constant size
    # def X(self, H: int, cols: Optional[List[str]], timestamps: Optional[List[int]] = None):
    #     if timestamps is None:
    #         timestamps = self.timestamps
    #     return [self.time_window(t-H, t, cols) for t in timestamps]

    # def y(self, F: int, cols: Optional[List[str]], timestamps: Optional[List[int]] = None):
    #     if timestamps is None:
    #         timestamps = self.timestamps
    #     return [self.time_window(t, t+F, cols) for t in self.timestamps]