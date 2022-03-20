import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Union, Optional

path = Path(r'.\trajectron-reproduction\data\pedestrians\eth\train\biwi_hotel_train.csv') # data source
path = Path('pedestrians/eth/train/biwi_hotel_train.txt', safe=False)






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
        self.H = 3
        self.F = 3
        self.data_cols   = ['t', 'id', 'x', 'y', 'vx', 'vy']
        self.input_cols  = ['x', 'y', 'vx', 'vy']
        self.output_cols = ['x', 'y']
        self.input_states = len(self.input_cols)
        self.output_states= len(self.output_cols)
        self.use_robot_node = False
        self.use_edge_nodes = True
        
        #### Load data in dataframe
        colnames = ['t', 'id', 'x', 'y']
        with open(path) as infile:
            df = pd.read_csv(infile, sep='\t', names=colnames)
            
        # convert time to seconds
        df['t'] = df['t']/10
        
        ids = df['id'].unique()
    
        for id in ids:
            # select rows for a specific id
            rows = df.loc[df['id'] == id]
            # add delta position between consecutive frames
            dt = 1
            df.loc[rows.index, 'vx'] = rows['x'].diff() / dt
            df.loc[rows.index, 'vy'] = rows['y'].diff() / dt
        # replace NaN elements with zero
        df = df.fillna(0)
        
        self.data = df
        

    
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
        batch : [x_i:           seq_H x 1 x input_states

                 x_neighbours:  seq_H x N x input_states

                 x_R:           seq_H x 1 x states

                 x_i_fut:       seq_F x 1 x input_states
                 
                 y_i:           seq_F x 1 x output_states]

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
                #TODO normalize neighbour data
                x_neighbours.append(self.time_window(t-(self.H+1), t, self.input_cols, id=neighbour))
                
                #TODO elemtwise sum as agregation operation
            x_neighbours = np.array(x_neighbours) #TODO: convert to right shape, currently: N x seq_H x states
            
            
            
        x_i = self.time_window(t-(self.H+1), t, self.input_cols, id=id)
        x_i_fut = self.time_window(t, t+self.F, self.input_cols, id=id)
        y_i = self.time_window(t, t+self.F, self.output_cols, id=id)
        
        # TODO: convert arrays to pytorch tensors and reshape
        
        
        return x_i, x_i_fut, y_i, x_R, x_neighbours
        
    def get_batches(self):
        
        batches = 0
        return batches

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
    
    









scene = Scene(path, header=0)
neigbours = scene.get_neighbours(id = 1, t = 0, include_node_i = False)
batch = scene.get_batch(3, 6)
print(batch)


