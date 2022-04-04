# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:34:23 2022

@author: maart
"""
# from preprocessing import import_ped_data, get_node_batch_data, get_node_batches
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from pathlib import Path

path = Path('data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)
df = import_ped_data(path)

def evaluate(scene, model):
    
    # set model to evaluation mode
    model.eval()
    
    X, _, Y_true, _, _ = scene.batch
    Y_pred = model(X)
    
    DE = abs(Y_true - Y_pred)# displacement error (l2 distance) #TODO: fix l2 distance
    ADE = 0# average displacement error
    FDE = 0# final displacement error
    
    return ADE, FDE



def evaluate_model(model, scene, node, t_min=1, t_max = None):
    """
    Evaluates dataset in df format and returns the evaluated df.

    Parameters
    ----------
    model : Trained model
    df : Dataset to be evaluated
    node: Node to be evaluated
    t_min : start time. The default is 1.
    t_max : end time.   The default is None.
    
    Returns
    -------
    df_eval

    """
    # set model to evaluation mode
    model.eval()
    
    
    x_i, x_i_fut, y_i, x_R, x_neighbours = scene.batch
    # X, Y_true = get_node_batches(df, node, model.H, model.F)
    X = x_i
    Y_true = y_i
    
    X = torch.reshape(torch.tensor(X),(model.H, -1, model.num_states))
    X = X.type(torch.FloatTensor)
    
    Y_true = torch.reshape(torch.tensor(Y_true),(model.H, -1, model.num_states))
    Y_true = Y_true.type(torch.FloatTensor)
    
    T = X.shape[1]

    Y_pred = model(X)
    Y_pred = Y_pred.detach().numpy().reshape((model.H, -1, model.num_states))
    
    # evaluate with true labels
    # Y_pred = Y_true
    
    plt.figure()
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    # flatten input X and Y to plot all points (including H and F points in seq)
    X = X.reshape((-1,2))
    # Y_pred = Y_pred.reshape((-1,2)) # TODO: this is wrong
    t_pred = 1
    Y_pred = Y_pred[t_pred,:,:].reshape((-1,2))

    plt.plot(X[:, 0], X[:, 1], c = 'b', label = 'True path', lw=2)
    plt.plot(Y_pred[:,0], Y_pred[:,1], c='red', label = 'Predicted path', lw = 0.5)
    plt.scatter(Y_pred[:,0], Y_pred[:,1], c='red')
    
    plt.legend()


# def plot_traj(df_true,df_pred, t_min=1, t_max = None):
#     """
#     Plot true and predicted trajectories
#     ----------
#     df_true : data frame with true trajectory
#     df_pred : data frame with predicted trajectory
#     t_min : start time. The default is 1.
#     t_max : end time.   The default is None.

#     Returns
#     -------
#     None.

#     """
#     if not(t_max == None):
#         df_true = df_true.loc[df_true['t'] >= t_min]
#         df_true = df_true.loc[df_true['t'] <= t_max] 
#         df_pred = df_pred.loc[df_pred['t'] >= t_min]
#         df_pred = df_pred.loc[df_pred['t'] <= t_max]
        
#     # assert (len(df_true)==len(df_pred))
#     nodes_true = np.unique(df_true['id'].values)
#     nodes_pred = np.unique(df_pred['id'].values)

#     fig = plt.figure()
#     plt.ylabel('y (m)')
#     plt.xlabel('x (m)')

#     for node in nodes_true:
#         # plot traj_true
#         df_i = df_true.loc[df_true['id'] == node]
#         x = df_i['x'].values
#         y = df_i['y'].values
#         plot_true = plt.plot(x,y, 'b', label = 'true')

#         # plot node number (beginning of traj, only for true, but starting point is the same)
#         plt.text(x[0], y[0], int(node))
        
#     for node in nodes_pred:
#         # plot traj_pred
#         df_i = df_pred.loc[df_pred['id'] == node]
#         noise = np.random.standard_normal(len(df_i))/20 # TODO fix real data
#         x = df_i['x'].values + noise
#         y = df_i['y'].values + noise
#         plot_pred = plt.plot(x,y, 'r--', label = 'pred')
        
def plot_node(df, i):
    plt.figure()
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    df_i = df.loc[df['id'] == int(i)]
    plt.plot(df_i['x'],df_i['y'])
    

plot_node(df, 11)
plot_traj(df, df, 0, 40)

