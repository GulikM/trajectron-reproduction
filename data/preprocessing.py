import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from itertools import product

# FRAME_FREQUENCY = 10 # [1/s] 
# # use to convert consecutive timestamps into delta time [s]

# # data source
# #inpath = pathlib.Path(r'./data\pedestrians\eth\train\biwi_hotel_train.txt')
# inpath = pathlib.Path('C:/Users/maart/Documents/GitHub/Trajectron-reproduction/data/pedestrians/eth/train/biwi_hotel_train.txt')
# # headers
# colnames = ['t', 'id', 'x', 'y']


# with open(inpath) as infile:
#     df = pd.read_csv(infile, sep='\t', names=colnames)
    
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
#         df.loc[rows.index, 'dt'] = rows['t'].diff() / FRAME_FREQUENCY
#         # convert delta position to velocity
#         df['dx'] /= df['dt']
#         df['dy'] /= df['dt']

#     # replace NaN elements with zero
#     df = df.fillna(0)
#     # fix time
#     df['t'] = df['t']/10
#     # note: velocities for the agents that just arrived in the scene are set to zero,
#     # even thought there is no way for us to know what the position of the agent was
#     # before it arrived on the scene.

#     # save
#    # df.to_csv(inpath.parent / (inpath.stem + '.csv'))
   
   


def import_ped_data(path, safe=False):
    colnames = ['t', 'id', 'x', 'y']
    with open(inpath) as infile:
        df = pd.read_csv(infile, sep='\t', names=colnames)
        
    # convert time to seconds
    df['t'] = df['t']/10
    
    if safe:
        df.to_csv(index=False)
    
    return df
    
inpath = pathlib.Path('C:/Users/maart/Documents/GitHub/Trajectron-reproduction/data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)
df = import_ped_data(inpath)

    

def plot_node(df, i):
    df_i = df.loc[df['id'] == int(i)]
    plt.scatter(df_i['x'],df_i['y'], s=df_i['t'])
    
for i in range(10):
    plot_node(df,i)




def get_node_batch_data(df, node, t, H, F):
    """
    Generate training data X and y for timestep t + history/future:
    X: (H+1)xD, with H timesteps in history and D node states
    y: (F+1)xD, with F timesteps in future and D node states

    Returns batch_X, batch_y

"""
    D = df.shape[1] # dimensions state space, should be 4
    
    df_node = df.loc[df['id'] == node] # data for node i
    df_seq_X = df_node.loc[df_node['t'] > t-H]  # data of node i for sequence [t-H,t]
    df_seq_X = df_seq_X.loc[df_seq_X['t'] <= t]  
    df_seq_y = df_node.loc[df_node['t'] > t] # data of node i for sequence [t-H,t]
    df_seq_y = df_seq_y.loc[df_seq_y['t'] <= t + F]
     
    # states for seq_X --> batch_X
    dt = 1 # assume constant timestep for velocity calculation
    x = df_seq_X['x'].values
    y = df_seq_X['y'].values
    # xdot = df_seq_X['dx'].values/dt
    # ydot = df_seq_X['dy'].values/dt
    # batch_X = np.array([x, y, xdot, ydot]).T
    # ydot = df_seq_X['dy'].values/dt
    batch_X = np.array([x, y]).T
     
     # states for seq_y --> batch_y
    dt = 1 # assume constant timestep for velocity calculation
    x = df_seq_y['x'].values
    y = df_seq_y['y'].values
    # xdot = df_seq_y['dx'].values/dt
    # ydot = df_seq_y['dy'].values/dt
    # batch_y = np.array([x, y, xdot, ydot]).T
    batch_y = np.array([x, y]).T

    return batch_X, batch_y

H,F = 3, 3
for t in range(0, int(df['t'].values[-1] + 1)):
    for ID in range(1, int(df['id'].values[-1]+1)):
        X,y = get_node_batch_data(df,ID,t,H,F)
        
        if (len(X)==H and len(y)==F):
            print(X.shape)
            
for ID in range(1, int(df['id'].values[-1]+1)):
    X,y = get_node_batch_data(df,ID,t,H,F)
    



