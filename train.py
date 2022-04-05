# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:29:34 2022

@author: maart
"""

from preprocessing import Scene
from pathlib import Path
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



def train(scene, net, 
          SEED = 42, 
          DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          num_epochs = 100,
          plot = True):
    """
    

    Parameters
    ----------
    scene : scene object containing training data
    net : neural network to trian
    optimizer : optimizer used for training, e.g. SGD

    Returns
    -------
    None.

    """
    print('Training on',DEVICE)
    #### Preprocess data from scene object:
    X_i, X_i_fut, Y_i, X_neighbours, X_i_present = scene.get_batches()
    B = X_i.shape[1]
    #### Make output deterministic 
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    #### Train mdoel
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-2)
    net.train()
    losses_train = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred, M_ps, M_qs = net(X_i, X_neighbours, X_i_fut, Y_i)
        loss = net.loss_function(M_qs, M_ps, Y_i.view(B,1,2), y_pred.view(B,1,25,1,6))
        loss.backward()
        optimizer.step()
        losses_train.append(loss.item())
        print("Epoch: ", epoch," Loss: ", loss.item())
        
        #TODO: add validation data later as well

    if plot:
        plt.plot(losses_train)
        plt.xlabel('epochs')
        plt.ylabel('loss')
    
    




