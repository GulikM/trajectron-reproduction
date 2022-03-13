import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from itertools import product
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def import_ped_data(path, safe=False):
    colnames = ['t', 'id', 'x', 'y']
    with open(path) as infile:
        df = pd.read_csv(infile, sep='\t', names=colnames)
        
    # convert time to seconds
    df['t'] = df['t']/10
    
    if safe:
        df.to_csv(index=False)
    
    return df
    

def get_node_batch_data(df, node, t, H, F):
    """
    Generate training data X and y for timestep t + history/future:
    X: (H)xD, with H timesteps in history and D node states
    y: (F)xD, with F timesteps in future and D node states

    Returns batch_X, batch_y
    
    Seq x batch x states
    
    
    TODO: add velocities to data

    """
    D = df.shape[1] # dimensions state space, should be 4; #TODO: remove harcoding of D
    
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
    
def get_node_batches(df, node, H, F):
    """
    Collects training data (stacked batches) of node i 
    over all timesteps t with enough data

    Returns trainX, trainY

"""
    D = df.shape[1] # dimensions state space, should be 4 #TODO: remove harcoding of D
    
    trainX = []
    trainY = []
    
    df_node = df.loc[df['id'] == node] # data for node i
    trange  = df_node['t'].values
    for t in trange:
        batchX, batchY = get_node_batch_data(df, node, t, H, F)
        if (len(batchX)==H and len(batchY)==F):  
            trainX.append(batchX)
            trainY.append(batchY)
            
    return np.array(trainX), np.array(trainY)

def get_batches(df, H, F):
    """
    Collects training data (stacked batches) of all nodes
    
    returns: 
        tensor trainX = seq x batchsize x D
        tensor trainY = seq x batchsize x D

    """
    D = df.shape[1] # dimensions state space, should be 4. #TODO:  remove hardcoding of D
    D = 2
    # init iteration over nodes with zeros so we can concatenate the data in the loop
    X = np.zeros((1,H,D))
    Y = np.zeros((1,F,D))

    H_in = X.shape[2]
    
    for node in range(1, int(df['id'].values[-1]+1)):  
        trainX,trainY = get_node_batches(df, node, H=H, F=F)      
        batchsize = len(trainX)        
        if batchsize > 0:
            X = np.concatenate([X, trainX], axis = 0)
            Y = np.concatenate([Y, trainY], axis = 0)

    X = torch.reshape(torch.tensor(X),(H,-1,H_in))
    Y = torch.reshape(torch.tensor(Y),(F,-1,H_in))
    X = X.type(torch.FloatTensor)
    Y = Y.type(torch.FloatTensor)
    
    return X, Y









