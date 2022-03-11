# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:02:52 2022

@author: maart
"""
from preprocessing import import_ped_data
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random



inpath = pathlib.Path('C:/Users/maart/Documents/GitHub/Trajectron-reproduction/data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)
df = import_ped_data(inpath)

def plot_traj(df_true,df_pred, t_min=1, t_max = None):
    """
    Plot true and predicted trajectories
    ----------
    df_true : data frame with true trajectory
    df_pred : data frame with predicted trajectory
    t_min : start time. The default is 1.
    t_max : end time.   The default is None.

    Returns
    -------
    None.

    """
    if not(t_max == None):
        df_true = df_true.loc[df_true['t'] >= t_min]
        df_true = df_true.loc[df_true['t'] <= t_max] 
        df_pred = df_pred.loc[df_pred['t'] >= t_min]
        df_pred = df_pred.loc[df_pred['t'] <= t_max]
        
    assert (len(df_true)==len(df_pred))
    nodes_true = np.unique(df_true['id'].values)
    nodes_pred = np.unique(df_pred['id'].values)

    fig = plt.figure()
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')

    for node in nodes_true:
        # plot traj_true
        df_i = df_true.loc[df_true['id'] == node]
        x = df_i['x'].values
        y = df_i['y'].values
        plot_true = plt.plot(x,y, 'b', label = 'true')

        # plot node number (beginning of traj, only for true, but starting point is the same)
        plt.text(x[0], y[0], int(node))
        
    for node in nodes_pred:
        # plot traj_pred
        df_i = df_pred.loc[df_pred['id'] == node]
        noise = np.random.standard_normal(len(df_i))/20 # TODO fix real data
        x = df_i['x'].values + noise
        y = df_i['y'].values + noise
        plot_pred = plt.plot(x,y, 'r--', label = 'pred')

    
plot_traj(df, df, 0, 40)

 