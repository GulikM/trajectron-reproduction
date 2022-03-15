import pandas as pd
from pathlib import Path

# from itertools import product

# DELTA_TIMESTAMP = 0.1 # 10 milllis in between frames 

path = Path(r'.\trajectron-reproduction\data\pedestrians\eth\train\biwi_hotel_train.csv') # data source

# colnames = ['t', 'id', 'x', 'y'] # headers

# df = pd.read_csv(path, delimiter='\t', names=colnames)

# ids = df['id'].unique()
# for id in ids:
#     # select rows for a specific id
#     rows = df.loc[df['id'] == id]
#     # add delta position between consecutive frames
#     df.loc[rows.index, 'dx'] = rows['x'].diff()
#     df.loc[rows.index, 'dy'] = rows['y'].diff()
#     # add delta time between consecutive frames
#     df.loc[rows.index, 'dt'] = rows['t'].diff() * 0.1
#     # convert delta position to velocity
#     df['dx'] /= df['dt']
#     df['dy'] /= df['dt']

# with open(path) as f:
#     df = pd.read_csv(f, delimiter='\t', names=colnames)
    
#     timestamps = df['t'].unique() 
#     ids = df['id'].unique()

#     idxs = list(product(timestamps, ids))
    
#     df = df.set_index(['t', 'id']).reindex(idxs).reset_index()

#     for id in ids:
#         # select rows for a specific id
#         rows = df.loc[df['id'] == id]
#         # add delta position between consecutive frames
#         df.loc[rows.index, 'dx'] = rows['x'].diff()
#         df.loc[rows.index, 'dy'] = rows['y'].diff()
#         # add delta time between consecutive frames
#         df.loc[rows.index, 'dt'] = rows['t'].diff() * DELTA_TIMESTAMP
#         # convert delta position to velocity
#         df['dx'] /= df['dt']
#         df['dy'] /= df['dt']

#     # replace NaN elements with zero
#     df = df.fillna(0)
#     # note: velocities for the agents that just arrived in the scene are set to zero,
#     # even though there is no way for us to know what the position of the agent was
#     # before it arrived on the scene.

#     # save
# df.to_csv(path.parent / (path.stem + '.csv'), index=False)





from typing import List, Union, Optional


class NodeType(object):
    def __init__(self, name: str) -> None:
        self.name = name
        # self.perception_range = perception_range

    def __str__(self):
        return self.name


class Node(object):
    instances = set()

    def __init__(self, type: NodeType, id: int, data: Optional[pd.DataFrame], is_robot: bool = False) -> None:
        self.type = type
        self.id = id
        self.data = data
        self.is_robot = is_robot

        self.timesteps = data['t'].unique()

        Node.instances.add(self)

    @classmethod
    def get(cls, key: str, value):
        for inst in cls.instances:
            if getattr(inst, key) == value:
                return inst

    def time_window(self, lower: int, upper: int, cols: List[str]): # TODO: add inclusive / exclusive slicing capabilities
        '''
        Returns a list of 

        :param upper: Upper bound index (inclusive) 
        :param lower: Lower bound index (exclusive)
        :param cols: 
        '''
        subset = self.data[cols]
        mask1 = subset['t'] >  lower
        mask2 = subset['t'] <= upper
        mask = mask1 * mask2
        return subset.loc[mask].to_numpy()
    
    def X(self, H: int, cols: List[str]):
        return [self.time_window(t-H, t, cols) for t in self.timesteps]

    def y(self, F: int, cols: List[str]):
        return [self.time_window(t, t+F, cols) for t in self.timesteps]
    
    
class Scene(object):
    '''
    
    '''
    def __init__(self, path: Union[str, Path], sep: str = ',', header: Optional[Union[int, List[int]]] = 0) -> None:
        '''
        :param path: 
        :param sep: Delimiter to use
        :param header: Row number(s) to use as the column names 
        '''
        if not isinstance(path, Path):
            path = Path(path)

        self.name = path.stem
        self.data = pd.read_csv(path, sep=sep, header=header) # TODO: enable other file formats and add reading in chunks
        self.nodes = []

        self.X_cols = None
        self.y_cols = None
        self.H = None
        self.F = None

    
    # def read_csv(self, path: Union[str, Path], sep: str = ',', header: Optional[Union[int, List[int]]] = None, chunksize: Optional[int] = None) -> pd.DataFrame:
    #     chunks = []
    #     for chunk in pd.read_csv(path, sep=sep, header=header, chunksize=chunksize):
    #         chunks.append(chunk)
    #     return pd.concat(chunks)


    def get_node_by_id(self, id: int) -> Node:
        for node in self.nodes:
            if node.id == id:
                return node

    def add_node_from_data(self, type: NodeType, id: int, is_robot: bool = False) -> None:
        mask = self.data['id'] == id
        node_data = self.data.loc[mask]
        self.nodes.append(Node(
            type=type,
            id=id, 
            data=node_data,
            is_robot=is_robot
        ))

    def add_nodes_from_data(self, ids: Optional[List[int]] = None) -> None:
        # If ids not specified, add all nodes
        if ids is None:
            ids = self.data['id'].unique()
        for id in ids:
            self.add_node_from_data(type=pedestrian, id=id) # TODO: make type assignment dynamic

    def remove_node(self, node: Node) -> None:
        self.nodes.remove(node)

    def set_X_cols(self, cols: List[str]) -> None:
        self.X_cols = cols

    def set_y_cols(self, cols: List[str]) -> None:
        self.y_cols = cols

    def set_H(self, H: int) -> None:
        self.H = H

    def set_F(self, F: int) -> None:
        self.F = F

    @property
    def X(self):
        import numpy as np
        self.nodes = [self.get_node_by_id(100)]
        tmp = [node.X(self.H, self.X_cols) for node in self.nodes]
        tmp = np.array(tmp)
        print(tmp.shape)
        return [node.X(self.H, self.X_cols) for node in self.nodes]

    @property
    def y(self): 
        return [node.y(self.F, self.y_cols) for node in self.nodes]








pedestrian = NodeType('pedestrian')
scene = Scene(path, header=0)

scene.add_nodes_from_data()
scene.set_X_cols(['x', 'y', 'dx', 'dy'])
scene.set_y_cols(['dx', 'dy'])
scene.set_H(20)
scene.set_F(10)

node = scene.get_node_by_id(100)
# node.X(100, ['t', 'id'])

scene.X


# class IDK(object):
#     '''
    
#     '''
#     def __init__(self, path: str, X_cols: List[str], y_cols: List[str]) -> None:        
#         df = pd.read_csv(path, header=0)
#         self.X = df[X_cols]
#         self.y = df[y_cols]

#     @property
#     def X(self):
#         return self.X

#     @property
#     def y(self):
#         return self.y    
    
#     def get_node_timestep_data(self, id: int, t: int, H: int, F: int):
#         '''
        
#         '''
#         X = self.X
#         y = self.y
    
#         X_node = X.loc[X['id'] == id]
#         X_batch = X_node.loc[X_node['t'] > t-H]
#         X_batch = X_batch.loc[X_batch['t'] <= t]
        
#         y_node = y.loc[y['id'] == id]
#         y_batch = y_node.loc[y_node['t'] > t]
#         y_batch = y_batch.loc[y_batch['t'] <= t+F]

#         return X_batch, y_batch

#     def get_node_data(self, node: int, H, F):
#         timestamps = self.X['t'].unique()
#         return [self.node_batch(node, t, H, F) for t in timestamps]

#     def get_data(self, t: int, H, F):
#         nodes = self.X['id'].unique()
#         return [self.batch(node, t, H, F) for node in nodes]

#     # def scatterplot(self, ids: List[int]) -> None:
#     #     for id in ids:
#     #         self.plot_node(id)

#     # def scatterplot_node(self, id: int, scaling: int == 1) -> None:
#     #     '''
#     #     Args: 
#     #             id: node id
#     #             scaling:
#     #     '''
#     #     X_node = self.X.loc[self.X['id'] == id]
#     #     plt.scatter(X_node['x'], X_node['y'], s=(scaling*self.X['t']))





# data_eth = IDK(path, X_cols=['t', 'id', 'x', 'y', 'dx', 'dy'], y_cols=['t', 'id', 'dx', 'dy'])