from typing import List, Optional

from src.data import CSVDataset
from src.environment import Node, pedestrian


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
    #         return self._dataset.data.__getitem__(key)

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

    def get_node(self, id: int) -> Node:
        for node in self._nodes:
            if node.id == id:
                return node
        raise NotImplementedError  # catchall

    def add_node(self, id: int) -> None:
        mask = self.ids == id
        node_data = self.data.loc[mask]
        node = Node(
            type=pedestrian,  # TODO: add dynamic type allocation
            id=id,
            data=node_data,
        )
        self._nodes.append(node)

    def add_nodes(self, ids: Optional[List[int]]):
        '''
        '''
        if ids is None:
            ids = self.ids
        for id in ids:
            self.add_node(id)

    def remove_node(self, id: int) -> None:
        node = self.get_node(id)
        self._nodes.remove(node)

    @property
    def robot(self) -> Node:
        for node in self._nodes:
            if node.is_robot:
                return node

    @robot.setter
    def robot(self, id: int) -> None:
        # deassign current robot
        bot = self.robot
        bot.is_robot = False
        # assign new robot
        node = self.get_node(id)
        node.is_robot = True

    def get_neighbors(self, id: int, timestamp: int):
        '''
        Return states of all nodes within perception range of node at timestamp.

        Args:
            id: 
            timestamp: 
        '''
        node = self.get_node(id)

        if not timestamp in node.timestamps:
            raise NotImplementedError  # TODO: add appropriate exception

        data = self._dataset.filter(row_filters={
            't': timestamp
        })

        neighbor_ids = self.ids.pop(id)

        subset = self._dataset.filter(row_filters={'t': [timestamp]})
        # separate node data from the neighbors
        nodedata = subset[subset['id'] == id]

    def get_neighbors(self, node_id: int, timestamp: int, include_node_i: bool = False):
        '''
        Method for 

        Args:
            node: Node for which all neighbors are returned
        '''

        node = self.get_node(node_id)

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

    # def get_batch(self, id, t):
    #     """
    #     Return batch for node i and time t

    #     Parameters
    #     ------
    #     id
    #     t
    #     Returns
    #     -------
    #     batch : [x_i:           seq_H x input_states
    #              x_neighbours:  seq_H x input_states (aggregated)
    #              x_R:           seq_H x states
    #              x_i_fut:       seq_F x input_states

    #              y_i:           seq_F x output_states]
    #     """
    #     x_R = []
    #     x_neighbours = []

    #     if self.use_robot_node:
    #         raise NotImplementedError
    #         x_R = []

    #     if self.use_edge_nodes:
    #         neighbours = self.get_neighbours(id = id, t = t)
    #         x_neighbours = []
    #         for neighbour in neighbours:
    #             #TODO normalize neighbour data (relative state + standardize)
    #             x_neighbour = self.time_window(t-(self.H+1), t, self.input_cols, id=neighbour)
    #             if len(x_neighbour)==self.H+1: #TODO: right now we only take into account neighbours with enoug data, but this does not have to be the case
    #                 x_neighbours.append(x_neighbour)

    #         x_neighbours = np.array(x_neighbours).reshape((-1, self.H+1, self.input_states)) 

    #         if self.aggregation_operation == 'sum':
    #             x_neighbours = np.sum(x_neighbours, axis=0)
    #         else:
    #             raise NotImplementedError

    #     x_i = self.time_window(t-(self.H+1), t, self.input_cols, id=id)
    #     x_i_fut = self.time_window(t, t+self.F, self.input_cols, id=id)
    #     y_i = self.time_window(t, t+self.F, self.output_cols, id=id)

    #     return x_i, x_i_fut, y_i, x_R, x_neighbours

    # def get_batches(self):
    #     """
    #     Iterate over all nodes and times and return batch data of scene
    #     Returns
    #     -------
    #     X_i : history of node i:                     seq_H+1 x N x input states.
    #     X_i_fut : future of node i:                  seq_F   x N x input states
    #     Y_i : label for node i:                      seq_F   x N x output states
    #     X_neighbours : Aggregated neighbour data:    seq_H+1 x N x input states
    #     """

    #     X_i         = torch.zeros((self.H+1, 1, self.input_states))
    #     X_i_fut     = torch.zeros((self.F, 1, self.input_states))
    #     Y_i         = torch.zeros((self.F, 1, self.output_states))
    #     X_neighbours= torch.zeros((self.H+1, 1, self.input_states))

    #     for id in self.ids:
    #         t_range = self.filter_data(id = id)['t'].values
    #         for t in t_range:
    #             x_i, x_i_fut, y_i, x_R, x_neighbours = self.get_batch(id, t) #TODO: make variable for if we use robot or not
    #             if (len(x_i)==len(x_neighbours)== self.H+1 and len(x_i_fut)==len(y_i)==self.F): # only store data if sequence long enough

    #                 ### convert to pytorch tensor and reshape:
    #                 x_i          = torch.tensor(x_i).reshape((self.H+1, 1, self.input_states))
    #                 x_neighbours = torch.tensor(x_neighbours).reshape((self.H+1, 1, self.input_states))
    #                 y_i          = torch.tensor(y_i).reshape((self.F, 1, self.output_states))
    #                 x_i_fut      = torch.tensor(x_i_fut).reshape((self.F, 1, self.input_states))     

    #                 X_i         = torch.cat((X_i, x_i), dim=1)
    #                 X_i_fut     = torch.cat((X_i_fut, x_i_fut), dim=1)
    #                 Y_i         = torch.cat((Y_i, y_i), dim=1)
    #                 X_neighbours= torch.cat((X_neighbours, x_neighbours), dim=1)

    #     return X_i, X_i_fut, Y_i, X_neighbours


path = r'trajectron-reproduction\data\pedestrians\eth\train\biwi_hotel_train.csv'
dataset = CSVDataset(path)
dataset.load(header=0)
dataset.validate()
scene = Scene(dataset)
scene.add_node_from_dataset()
scene.filter(row_filters={'haha': 0, 'hihi': 1})
