# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:46:25 2022

@author: maart

Main script to run:
    Trains model on training data
    Evaluates on test data
    Visualizes some results
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
from pathlib import Path
from preprocessing import Scene
from train import train

#### Define dataset and make scene instances
train_path = Path('data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)
train_scene = Scene(train_path, header=0)

test_path = Path('data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)
test_scene = Scene(test_path, header=0)

#### Init model and train on data
net = model()
train(train_scene, net)

#### Evaluate model on test data and visualize results
# test(test_scene, net)
















