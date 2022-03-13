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



inpath = pathlib.Path('C:/Users/maart/Documents/GitHub/Trajectron-reproduction/data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)
df = import_ped_data(inpath)

def plot_data(df,t_min = None,t_max= None, node = None):
    
    if not(node == None):
        df = df.loc[df['id'] == node]
    
    plt.figure
    
    print(df)
    df = df.loc[df['t'] >= t_min]
    df = df.loc[df['t'] <= t_max]    
    # plt.scatter(df['x'],df['y'])
    plt.scatter(df['x'],df['y'], s = 10*df['t']/(1+t_min), c=df['id'])
        
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    plt.show()

    

plot_data(df, 0, 2000, node=31)

 