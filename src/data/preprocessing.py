import pandas as pd
from pathlib import Path

import numpy as np
import torch
from load import CSVDataset
from functools import partial

# from itertools import product

# DELTA_TIMESTAMP = 0.1 # 10 milllis in between frames 

path = Path(r'.\data\pedestrians\eth\train\biwi_hotel_train.txt') # data source

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
import inspect




class NodeType(object):
    def __init__(self, name: str) -> None:
        self.name = name
        # self.perception_range = perception_range

    def __str__(self):
        return self.name


class Node(object):
    # instances = set()

    def __init__(self, type: Optional[NodeType], id: int, data: Optional[pd.DataFrame], is_robot: bool = False) -> None:
        self.type = type
        self.id = id
        self._data = data
        self.is_robot = is_robot

        # Node.instances.append(self)

    # @classmethod
    # def get(cls, key: str, value):
    #     for inst in cls.instances:
    #         if getattr(inst, key) == value:
    #             return inst

    # def get_neighbors(self, id: int):
    #     return [node for node in self.__class__.instances if node.id != id]
    
    @property
    def timestamps(self):
        return self._data['t'].unique()
    
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




class Scene(object):
    def __init__(self, dataset) -> None:
        if not isinstance(dataset, CSVDataset):
            raise TypeError(f'{dataset} must be {CSVDataset} instance')
        self._dataset = dataset
        self._nodes = []

    # def __getitem__(self, key: str):
    #     if key == "t":
    #         return self.timestamps()
    #     elif key == "id":
    #         return self.ids()
    #     else:
    #         return self._dataset.__getitem__(key)

    @property
    def data(self):
        return self._dataset.data

    @property
    def name(self):
        return self._dataset.name

    @property
    def timestamps(self):
        return self.data['t'].unique()

    @property
    def ids(self):
        return self.data['id'].unique()

    def get_node_by_id(self, id: int) -> Node:
        for node in self._nodes:
            if node.id == id:
                return node

    def add_node_from_dataset(self, ids: Optional[Union[List[int], int]] = None):
        ''' 
        '''
        if ids is None:
            ids = self.ids
        if isinstance(ids, int):
            ids = [ids]
        print(self.data.head())
        candidate_nodes = self.data.loc[ids]
        # TODO: figure out pandas mapping
        for id in ids:
            print(self.data.loc[id])
            mask = self.ids == id
            node_data = self.data.loc[mask]
            node = Node(
                type=None,  # TODO: add dynamic type allocation
                id=id,
                data=node_data,
            )
            self._nodes.append(node)

    def remove_node(self, id: int) -> None:
        node = self.get_node_by_id(id)
        self._nodes.remove(node)

    @property
    def robot(self) -> Node:
        for node in self._nodes:
            if node.is_robot:
                return node

    @robot.setter
    def robot(self, id: int) -> None:
        robot = self.robot
        robot.is_robot = False
        node = self.get_node_by_id(id)
        node.is_robot = True

    # def X(self, H: int, cols: Optional[List[str]]):
    #     return [node.X(H, cols) for node in self.nodes]

    # def y(self, F: int, cols: Optional[List[str]]): 
    #     return [node.y(F, cols) for node in self.nodes]


    def filter(self, timestamps: Optional[List[int]] = None, ids: Optional[List[int]] = None, columns: Optional[List[str]] = None):
        '''
        Method for 
        
        :param timestmaps:
        :param ids: 
        :param cols:       
        :return:  
        '''
        if timestamps is None:
            timestamps = self.timestamps
        if ids is None:
            ids = self.ids
        if columns is None:
            columns = self._dataset.header       

        mask1 = self.data['t'].isin(timestamps) 
        mask2 = self.data['id'].isin(ids)        
        rows  = mask1 & mask2
        return self.data.loc[rows, columns].to_numpy()

    def get_neighbors(self, node_id: int, timestamp: int, include_node_i: bool = False):
        node = self.get_node_by_id(node_id)
        print(node)

            # if timestamp not in node.timestamps:
            #     raise NotImplementedError
            
            # # get all relevent node data
            # data = self.filter(timestamps=list(timestamp))
            # print(data)

        

        


    # def get_neighbours(self, id: int, t: int, include_node_i: bool = False):
    #     """
    #     Returns an array with the neighbour nodes of node_i wihtin the perception range at time t
    #     Parameters
    #     ----------
    #     Args:
    #         id : node id.
    #         timestamp : timestamp.
    #     Returns
    #     -------
    #     neighbours : array with neighbour nodes
    #     """
        
        


    #     df_node_i = self.filter_data(id=id, t = t)
    #     df_nodes  = self.filter_data(t = t)
        
    #     if (len(df_node_i)==0 or len(df_nodes)==0):
    #         print('No data available for given t (and node id)')
    #         neighbours = np.array([])
    #     else:      
    #         pos_node_i = np.array([df_node_i['x'].values * np.ones(len(df_nodes)), 
    #                                df_node_i['y'].values * np.ones(len(df_nodes))])
    #         pos_nodes  = np.array([df_nodes['x'], df_nodes['y']])
    #         distances  = np.linalg.norm(pos_nodes - pos_node_i, axis = 0)
    #         perception_logic = (distances <= self.attention_radius)
    #         not_node_i = (distances != 0) 
    #         if include_node_i:
    #             not_node_i = True
    #         neighbours = df_nodes['id'][not_node_i * perception_logic].values 
        
    #     return neighbours


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






path = r'pedestrians\eth\train\biwi_hotel_train.csv'
dataset = CSVDataset(path)
dataset.load(header=0)
dataset.validate()
scene = Scene(dataset)
scene.add_node_from_dataset()
scene.get_neighbors(1, 1600)




# class IDK(object):

#     def __init__(self, model) -> None:
#         if not isinstance(model, nn.Module):
#             raise NotImplementedError
#         self._model = model


#         # add device / cuda AND model = model.to(device)



#     def __call__(self, x):
#         # simple forward pass of the model with input x: return y = self._model(x)
#         pass

    


#     def train(train_loader, net, optimizer, criterion):
#         """
#         Trains network for one epoch in batches.

#         Args:
#             train_loader: Data loader for training set.
#             net: Neural network model.
#             optimizer: Optimizer (e.g. SGD).
#             criterion: Loss function (e.g. cross-entropy loss).
#         """
    
#         avg_loss = 0
#         correct = 0
#         total = 0

        # # iterate through batches
        # for i, data in enumerate(train_loader):
        #     # get the inputs; data is a list of [inputs, labels]
        #     inputs, labels = data

        #     # zero the parameter gradients
        #     optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # keep track of loss and accuracy
    #         avg_loss += loss
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #     return avg_loss/len(train_loader), 100 * correct / total


    # def train(self, train_loader, optimizer, criterion, device):
    #     pass

    # def test():
    #     pass

    # def evaluate():
    #     pass

    # def run(self):
    #     pass