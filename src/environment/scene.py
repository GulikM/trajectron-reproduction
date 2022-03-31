from typing import List, Optional

import numpy as np
import torch

from src.data import CSVDataset
from src.environment import Node, pedestrian

class Scene(object):
    def __init__(self,
                 dataset,
                 use_edge_nodes: bool = True,
                 use_robot_node: bool = False,
                 H: int = 30,
                 F: int = 3,
                 input_columns = None,
                 output_columns = None,
                 aggregation_operation: str = 'sum'
    ) -> None:
        if not isinstance(dataset, CSVDataset):
            raise TypeError(f'{dataset} must be {CSVDataset} instance')
        # data
        self._dataset = dataset
        self._nodes = []
        # hyperparameters
        self.use_edge_nodes = use_edge_nodes
        self.use_robot_node = use_robot_node
        self.H = H
        self.F = F
        self.input_columns = input_columns or ['x', 'y', 'dx', 'dy']
        self.output_columns = output_columns or ['x', 'y']
        self.aggregation_operation = aggregation_operation

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
        mask = self.data['id'] == id
        node_data = self.data.loc[mask]
        node = Node(
            type=pedestrian,  # TODO: add dynamic type allocation
            id=id,
            data=node_data,
        )
        self._nodes.append(node)

    def add_nodes(self, ids: Optional[List[int]] = None):
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

    def get_neighbors(self, id: int, timestamp: int, include_node: bool = False):
        '''

        Args:
            id:
            timestamp:
            include_node:
        '''
        node = self.get_node(id)
        if not timestamp in node.timestamps:
            raise KeyError(f'{timestamp} not in {node.timestamps}')

        data = self._dataset.filter(
            columns=['id', 'x', 'y'],
            row_filters={
                't': [timestamp]
            }
        )

        mask = data['id'] == id
        # split dataframe
        node_data = data[mask]
        neighbors_data = data[~mask]

        relative_positions = neighbors_data.subtract(node_data.values) # distances between neighbors and node
        straight_line_distances = np.linalg.norm(relative_positions, axis=1) # 2-norm
        in_perception_range = straight_line_distances <= node.type.perception_range
        # get neighbors ids for neighbors in perception range
        neighbor_ids = neighbors_data['id'][in_perception_range].tolist()
        # if specified, add node id to neighbours list
        if include_node:
            neighbor_ids.append(id)
        # cast ids to int
        return list(map(int, neighbor_ids))

    def get_batch(self, id: int, timestamp: int):
        """ # TODO: clarify what states and output_states mean
        Returns a batch for node {id} at time {timestamp}

        batch : [x_i:           H x len(input_columns)
                 x_i_fut:       F x len(input_columns)
                 y_i:           F x output_states
                 x_R:           H x states
                 x_neighbours:  H x len(input_columns) (aggregated)]
        """
        history_timestamps = np.arange(timestamp - self.H + 1, timestamp)
        future_timestamps = np.arange(timestamp, timestamp + self.F)

        x_R = []
        if self.use_robot_node:
            raise NotImplementedError

        x_neighbors = []
        if self.use_edge_nodes:
            neighbor_ids = self.get_neighbors(id, timestamp)
            for neighbor_id in neighbor_ids:
                # TODO normalize neighbour data (relative state + standardize)
                x_neighbor = self._dataset.filter(
                    columns=self.input_columns,
                    row_filters={
                        't': history_timestamps,
                        'id': [neighbor_id]
                    }
                )
                if len(x_neighbor) == self.H+1:
                    x_neighbors.append(x_neighbor)
                    # Note that now we only take into account the neighbors which
                    # are in the perception range of the node during the whole time window.
                    # This does not necessarily have to be the case

            x_neighbors = np.array(x_neighbors).reshape((-1, self.H+1, len(self.input_columns)))

            if self.aggregation_operation == 'sum':
                x_neighbors = np.sum(x_neighbors, axis=0)
            else:
                raise NotImplementedError

        x_i = self._dataset.filter(
            columns=self.input_columns,
            row_filters={
                't': history_timestamps,
                'id': [id]
            }
        )

        x_i_fut = self._dataset.filter(
            columns=self.input_columns,
            row_filters={
                't': future_timestamps,
                'id': [id]
            }
        )

        y_i = self._dataset.filter(
            columns=self.output_columns,
            row_filters={
                't': future_timestamps,
                'id': [id]
            }
        )

        return x_i, x_i_fut, y_i, x_R, x_neighbors

    def get_batches(self):
        """
        Iterates over all nodes and timestamps and returns batch data of scene
        -------
        X_i : history of node i:                     seq_H+1 x N x input states.
        X_i_fut : future of node i:                  seq_F   x N x input states
        Y_i : label for node i:                      seq_F   x N x output states
        X_neighbours : Aggregated neighbour data:    seq_H+1 x N x input states
        """

        X_i         = torch.zeros((self.H+1, 1, len(self.input_columns)))
        X_i_fut     = torch.zeros((self.F, 1, len(self.input_columns)))
        Y_i         = torch.zeros((self.F, 1, len(self.output_columns)))
        X_neighbours= torch.zeros((self.H+1, 1, len(self.input_columns)))

        for id in self.ids:
            node = self.get_node(id)
            node_timestamps = node.timestamps
            # t_range = self.filter_data(id = id)['t'].values
            # for t in t_range:
            for timestamp in node_timestamps:
                x_i, x_i_fut, y_i, x_R, x_neighbours = self.get_batch(id, timestamp) #TODO: make variable for if we use robot or not
                if (len(x_i)==len(x_neighbours)== self.H+1 and len(x_i_fut)==len(y_i)==self.F): # only store data if sequence long enough

                    # convert to pytorch tensor and reshape:
                    x_i          = torch.tensor(x_i).reshape((self.H+1, 1, len(self.input_columns)))
                    x_neighbours = torch.tensor(x_neighbours).reshape((self.H+1, 1, len(self.input_columns)))
                    y_i          = torch.tensor(y_i).reshape((self.F, 1, len(self.output_columns)))
                    x_i_fut      = torch.tensor(x_i_fut).reshape((self.F, 1, len(self.input_columns)))

                    X_i          = torch.cat((X_i, x_i), dim=1)
                    X_i_fut      = torch.cat((X_i_fut, x_i_fut), dim=1)
                    Y_i          = torch.cat((Y_i, y_i), dim=1)
                    X_neighbours = torch.cat((X_neighbours, x_neighbours), dim=1)

        return X_i, X_i_fut, Y_i, X_neighbours