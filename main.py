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
from train import train
from evaluate import evaluate

#### Define dataset and make scene instances
train_path = Path('data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)
train_scene = Scene(train_path, header=0)

val_path = Path('data/pedestrians/eth/val/biwi_hotel_val.txt', safe=False)
val_scene = Scene(train_path, header=0)

test_path = Path('data/pedestrians/eth/test/biwi_eth.txt', safe=False)
test_scene = Scene(test_path, header=0)

#### Preprocess data; init and train model
# input, output, history and future defined in scene object
train_scene.get_batches()

net = model(batch_size=train_scene.batch_size,
            input_size = train_scene.input_states, 
            History = train_scene.H, 
            Future = train_scene.F)

train(train_scene, net, num_epochs = 100)

#### Evaluate model on test data and visualize results
test_scene.get_batches()
ADE, FDE = evaluate(test_scene, net)











