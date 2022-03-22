# import pandas as pd
# import pathlib
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# from itertools import product
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# def import_ped_data(path, safe=False):
#     colnames = ['t', 'id', 'x', 'y']
#     with open(path) as infile:
#         df = pd.read_csv(infile, sep='\t', names=colnames)
        
#     # convert time to seconds
#     df['t'] = df['t']/10
    
#     if safe:
#         df.to_csv(index=False)
    
#     return df
    

# def get_node_batch_data(df, node, t, H, F):
#     """
#     Generate training data X and y for timestep t + history/future:
#     X: (H)xD, with H timesteps in history and D node states
#     y: (F)xD, with F timesteps in future and D node states

#     Returns batch_X, batch_y
    
#     Seq x batch x states
    
    
#     TODO: add velocities to data

#     """
#     D = df.shape[1] # dimensions state space, should be 4; #TODO: remove harcoding of D
    
#     df_node = df.loc[df['id'] == node] # data for node i
#     df_seq_X = df_node.loc[df_node['t'] > t-H]  # data of node i for sequence [t-H,t]
#     df_seq_X = df_seq_X.loc[df_seq_X['t'] <= t]  
#     df_seq_y = df_node.loc[df_node['t'] > t] # data of node i for sequence [t-H,t]
#     df_seq_y = df_seq_y.loc[df_seq_y['t'] <= t + F]
     
#     # states for seq_X --> batch_X
#     dt = 1 # assume constant timestep for velocity calculation
#     x = df_seq_X['x'].values
#     y = df_seq_X['y'].values
#     # xdot = df_seq_X['dx'].values/dt
#     # ydot = df_seq_X['dy'].values/dt
#     # batch_X = np.array([x, y, xdot, ydot]).T
#     # ydot = df_seq_X['dy'].values/dt
#     batch_X = np.array([x, y]).T
     
#      # states for seq_y --> batch_y
#     dt = 1 # assume constant timestep for velocity calculation
#     x = df_seq_y['x'].values
#     y = df_seq_y['y'].values
#     # xdot = df_seq_y['dx'].values/dt
#     # ydot = df_seq_y['dy'].values/dt
#     # batch_y = np.array([x, y, xdot, ydot]).T
#     batch_y = np.array([x, y]).T

#     return batch_X, batch_y
    
# def get_node_batches(df, node, H, F):
#     """
#     Collects training data (stacked batches) of node i 
#     over all timesteps t with enough data

#     Returns trainX, trainY

#     """
#     D = df.shape[1] # dimensions state space, should be 4 #TODO: remove harcoding of D
    
#     trainX = []
#     trainY = []
    
#     df_node = df.loc[df['id'] == node] # data for node i
#     trange  = df_node['t'].values
#     for t in trange:
#         batchX, batchY = get_node_batch_data(df, node, t, H, F)
#         if (len(batchX)==H and len(batchY)==F):  
#             trainX.append(batchX)
#             trainY.append(batchY)
            
#     return np.array(trainX), np.array(trainY)

# def get_batches(df, H, F):
#     """
#     Collects training data (stacked batches) of all nodes
    
#     returns: 
#         tensor trainX = seq x batchsize x D
#         tensor trainY = seq x batchsize x D

#     """
#     D = df.shape[1] # dimensions state space, should be 4. #TODO:  remove hardcoding of D
#     D = 2
#     # init iteration over nodes with zeros so we can concatenate the data in the loop
#     X = np.zeros((1,H,D))
#     Y = np.zeros((1,F,D))

#     H_in = X.shape[2]
    
#     for node in range(1, int(df['id'].values[-1]+1)):  
#         trainX,trainY = get_node_batches(df, node, H=H, F=F)      
#         batchsize = len(trainX)        
#         if batchsize > 0:
#             X = np.concatenate([X, trainX], axis = 0)
#             Y = np.concatenate([Y, trainY], axis = 0)

#     X = torch.reshape(torch.tensor(X),(H,-1,H_in))
#     Y = torch.reshape(torch.tensor(Y),(F,-1,H_in))
#     X = X.type(torch.FloatTensor)
#     Y = Y.type(torch.FloatTensor)
    
#     return X, Y

import pandas as pd
from pathlib import Path
import numpy as np
import torch
from typing import List, Union, Optional

# path = Path(r'.\trajectron-reproduction\data\pedestrians\eth\train\biwi_hotel_train.csv') # data source
# path = Path('data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)






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
        #self.data = pd.read_csv(path, sep=sep, header=header) # TODO: enable other file formats and add reading in chunks
        self.nodes = []

        self.X_cols = None
        self.y_cols = None

        
        #### Load  hyperparameters:
        
        self.attention_radius = 5 # m (for pedestrians)
        self.H = 2
        self.F = 3
        self.data_cols   = ['t', 'id', 'x', 'y', 'vx', 'vy']
        self.input_cols  = ['x', 'y', 'vx', 'vy']
        self.output_cols = ['x', 'y']
        self.input_states = len(self.input_cols)
        self.output_states= len(self.output_cols)
        self.use_robot_node = False
        self.use_edge_nodes = True
        self.aggregation_operation = 'sum'

        #### Load data in dataframe
        colnames = ['t', 'id', 'x', 'y']
        with open(path) as infile:
            df = pd.read_csv(infile, sep='\t', names=colnames)
            
        # convert time to seconds
        df['t'] = df['t']/10
        
        ids = df['id'].unique()
        self.ids = ids
        for id in ids:
            # select rows for a specific id
            rows = df.loc[df['id'] == id]
            # add delta position between consecutive frames
            dt = 1
            df.loc[rows.index, 'vx'] = rows['x'].diff() / dt
            df.loc[rows.index, 'vy'] = rows['y'].diff() / dt
        # replace NaN elements with zero: 
        df = df.fillna(0)
        
        self.data = df
        self.batch = None
        

    
    def filter_data(self, id = None, t = None):
        """
        Fitlers data for given time t and node id
        
        Parameters
        ----------
        id : node id to filter, optional
             The default is None.
        t : time t to filter, optional
            The default is None.

        Returns
        -------
        filtered data

        """
        
        mask_id = True
        mask_t  = True
        
        if not(id==None):
            mask_id = self.data['id'] == id
            
        if not(t==None):
            mask_t  = self.data['t']  == t
 
        mask = mask_t * mask_id
        
        return self.data[mask]

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

    # def add_nodes_from_data(self, ids: Optional[List[int]] = None) -> None:
    #     # If ids not specified, add all nodes
    #     if ids is None:
    #         ids = self.data['id'].unique()
    #     for id in ids:
    #         self.add_node_from_data(type=pedestrian, id=id) # TODO: make type assignment dynamic

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
        
    # def time_window(self, upper_bound: int, lower_bound: int, cols: Optional[List[str]], nodes: Optional[List[Node]] = None):
    #    	if nodes is None:
    #    		nodes = self._nodes
    #    	return [node.time_window(upper_bound, lower_bound, cols) for node in nodes]
       
    def time_window(self, lower: int, upper: int, cols: List[str], id = None):
        '''
        Returns a list of 

        :param upper: Upper bound index (inclusive) 
        :param lower: Lower bound index (exclusive)
        :param cols: 
        '''
        
        if not(id==None):
            mask_id = self.data['id'] == id
        else:
            mask_id = True
            
        mask1 = self.data['t'] >  lower
        mask2 = self.data['t'] <= upper
        mask = mask1 * mask2 * mask_id
        
        
        subset = self.data[cols]

        return subset.loc[mask].to_numpy()
        
    def get_neighbours(self, id: int, t: int, include_node_i = False):
        """
        Returns an array with the neighbour nodes of node_i wihtin the perception range at time t

        Parameters
        ----------
        id : int
            node id.
        t : int
            time t.

        Returns
        -------
        neighbours : array with neighbour nodes

        """
        
        df_node_i = self.filter_data(id = id, t = t)
        df_nodes  = self.filter_data(t = t)
        
        if (len(df_node_i)==0 or len(df_nodes)==0):
            print('No data available for given t (and node id)')
            neighbours = np.array([])
        else:      
            pos_node_i = np.array([df_node_i['x'].values * np.ones(len(df_nodes)), 
                                   df_node_i['y'].values * np.ones(len(df_nodes))])
            pos_nodes  = np.array([df_nodes['x'], df_nodes['y']])
            distances  = np.linalg.norm(pos_nodes - pos_node_i, axis = 0)
            perception_logic = (distances <= self.attention_radius)
            not_node_i = (distances != 0) 
            if include_node_i:
                not_node_i = True
            neighbours = df_nodes['id'][not_node_i * perception_logic].values 
        
        return neighbours
    
        
    def get_batch(self, id, t):
        """
        Return batch for node i and time t
        
        Parameters
        ------
        id
        t

        Returns
        -------
        batch : [x_i:           seq_H x input_states

                 x_neighbours:  seq_H x input_states (aggregated)

                 x_R:           seq_H x states

                 x_i_fut:       seq_F x input_states
                 
                 y_i:           seq_F x output_states]

        """
        x_R = []
        x_neighbours = []
        
        if self.use_robot_node:
            raise NotImplementedError
            x_R = []
        
        if self.use_edge_nodes:
            neighbours = self.get_neighbours(id = id, t = t)
            x_neighbours = []
            for neighbour in neighbours:
                #TODO normalize neighbour data (relative state + standardize)
                x_neighbour = self.time_window(t-(self.H+1), t, self.input_cols, id=neighbour)
                if len(x_neighbour)==self.H+1: #TODO: right now we only take into account neighbours with enoug data, but this does not have to be the case
                    x_neighbours.append(x_neighbour)

            x_neighbours = np.array(x_neighbours).reshape((-1, self.H+1, self.input_states)) 

            if self.aggregation_operation == 'sum':
                x_neighbours = np.sum(x_neighbours, axis=0)
            else:
                raise NotImplementedError
 
        x_i = self.time_window(t-(self.H+1), t, self.input_cols, id=id)
        x_i_fut = self.time_window(t, t+self.F, self.input_cols, id=id)
        y_i = self.time_window(t, t+self.F, self.output_cols, id=id)
         
        self.batch = x_i, x_i_fut, y_i, x_R, x_neighbours
        
        return x_i, x_i_fut, y_i, x_R, x_neighbours
        
    def get_batches(self):
        """
        Iterate over all nodes and times and return batch data of scene

        Returns
        -------
        X_i : history of node i:                     seq_H+1 x N x input states.
        X_i_fut : future of node i:                  seq_F   x N x input states
        Y_i : label for node i:                      seq_F   x N x output states
        X_neighbours : Aggregated neighbour data:    seq_H+1 x N x input states

        """
        
        X_i         = torch.zeros((self.H+1, 1, self.input_states))
        X_i_fut     = torch.zeros((self.F, 1, self.input_states))
        Y_i         = torch.zeros((self.F, 1, self.output_states))
        X_neighbours= torch.zeros((self.H+1, 1, self.input_states))
        
        for id in self.ids:
            t_range = self.filter_data(id = id)['t'].values
            for t in t_range:
                x_i, x_i_fut, y_i, x_R, x_neighbours = self.get_batch(id, t) #TODO: make variable for if we use robot or not
                if (len(x_i)==len(x_neighbours)== self.H+1 and len(x_i_fut)==len(y_i)==self.F): # only store data if sequence long enough
                
                    ### convert to pytorch tensor and reshape:
                    x_i          = torch.tensor(x_i).reshape((self.H+1, 1, self.input_states))
                    x_neighbours = torch.tensor(x_neighbours).reshape((self.H+1, 1, self.input_states))
                    y_i          = torch.tensor(y_i).reshape((self.F, 1, self.output_states))
                    x_i_fut      = torch.tensor(x_i_fut).reshape((self.F, 1, self.input_states))     
                    
                    X_i         = torch.cat((X_i, x_i), dim=1)
                    X_i_fut     = torch.cat((X_i_fut, x_i_fut), dim=1)
                    Y_i         = torch.cat((Y_i, y_i), dim=1)
                    X_neighbours= torch.cat((X_neighbours, x_neighbours), dim=1)
                    
        return X_i, X_i_fut, Y_i, X_neighbours

    # @property
    # def X(self):
    #     import numpy as np
    #     self.nodes = [self.get_node_by_id(100)]
    #     tmp = [node.X(self.H, self.X_cols) for node in self.nodes]
    #     tmp = np.array(tmp)
    #     print(tmp.shape)
    #     return [node.X(self.H, self.X_cols) for node in self.nodes]

    # @property
    # def y(self): 
    #     return [node.y(self.F, self.y_cols) for node in self.nodes]
    
    


# pedestrian = NodeType('pedestrian')
# scene = Scene(path, header=0)
# scene.add_nodes_from_data()

# X_i, X_i_fut, Y_i, X_neighbours = scene.get_batches()











