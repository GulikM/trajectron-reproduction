# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:23:07 2022

@author: maart

Evaluate function
    Print metrics ADE, FDE
    Test loss
    Visualize results
"""

def evaluate(scene, net, plot = True):
    
    #### Preprocess data from scene object:
    X_i, X_i_fut, Y_i, X_neighbours, X_i_present = scene.get_batches()
    
    net.eval()
    y_pred, M_ps, M_qs = net(X_i, X_neighbours, X_i_fut, Y_i)
    loss = net.loss_function(M_qs, M_ps, Y_i, y_pred)
    
    return ADE, FDE, loss


def get_metrics():
    
    return ADE, FDE
