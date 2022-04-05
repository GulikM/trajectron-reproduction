# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:23:07 2022

@author: maart

Evaluate function
    Print metrics ADE, FDE
    Test loss
    Visualize results
"""
import matplotlib.pyplot as plt
def evaluate(scene, net, plot = True):
    
    #### Assign preprocessed data
    X_i = scene.X_i
    Y_i = scene.Y_i
    X_neighbours = scene.X_neighbours
    X_i_fut = scene.X_i_fut
    B = scene.batch_size
    net.batch_size = B

    net.eval()
    net.training = False
    y_pred, M_ps, M_qs = net(X_i, X_neighbours, X_i_fut, Y_i)
    
    # if net.training:
    #     loss = net.loss_function(M_qs, M_ps, Y_i.view(B,1,2), y_pred.view(B,1,25,1,6)).item()
    # else:
    #     loss = net.loss_function(M_qs, M_ps, Y_i.view(B,1,2), y_pred.view(B,1,1,1,6)).item()
    
    y_true = Y_i.reshape((-1, 2)).detach().numpy()
    y_pred = y_pred[:,:,:, 1:3].reshape(-1,2).detach().numpy()
    
    if plot:
        for B in range(10):
            plt.figure()
            plt.scatter(y_true[10*B:10*B+10,0], y_true[10*B:10*B+10,1], c='blue')
            plt.scatter(y_pred[10*B:10*B+10,0], y_pred[10*B:10*B+10,1], c='red')
    
    ADE = 0
    FDE = 0
    return ADE, FDE



