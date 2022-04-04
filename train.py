# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:29:34 2022

@author: maart
"""

from preprocessing import Scene
from pathlib import Path
from model_class import model
from unicodedata import bidirectional
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def train(scene, model, optimizer, 
          SEED = 42 
          DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          num_epochs = 100,
          learning_rate = 0.01):
    """
    

    Parameters
    ----------
    scene : scene object containing training data
    model : neural network to trian
    optimizer : optimizer used for training, e.g. SGD

    Returns
    -------
    None.

    """
    print('Training on',DEVICE)
    #### Preprocess data from scene object:
    X_i, X_i_fut, Y_i, X_neighbours, X_i_present = scene.get_batches()
    
    #### Make output deterministic 
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    #### Train mdoel
    for epoch in range(num_epochs):
        y_pred, M_ps, M_qs = net(X_i, X_neighbours, X_i_fut, Y_i)
        loss = net.loss_function(M_qs, M_ps, Y_i, y_pred)
        optimizer.step()
        losses_train.append(loss.item())
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        
        #TODO: add validation data later as well

    return
    
    




num_epochs = 1000
learning_rate = 0.01

input_size = 4
hidden_size = 32
num_layers = 1
History = 3
Future = 3
num_classes = 2
K_p = 25
N_p = 1
K_q = 25
N_q = 1

hidden_history = 32
hidden_interactions = 8
hidden_future = 32

batch_first = True

GRU_size = 128

# For debugging the forward function and model
# initialize model object
net = model(input_size, History, Future, hidden_history, hidden_interactions, hidden_future, GRU_size, batch_first, K_p, N_p, K_q, N_q)

# do forward function
y_pred, M_p_norm, M_q_norm = net.forward(X_i, X_neighbours, X_i_fut, Y_i)



for epoch in range(num_epochs):
    y_pred, M_p_norm, M_q_norm = net.forward(X_i, X_neighbours, X_i_fut, Y_i)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, Y_train)
    loss.backward()   
    optimizer.step()
    losses_train.append(loss.item())
    loss_val = criterion(lstm(X_val), Y_val)
    losses_val.append(loss_val.item())
    
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

