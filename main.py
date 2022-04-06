import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchsummary import summary
import torch.nn.functional as F
import random
import preprocessing as pre
from model_class import model
from loss import loss_function


# Use gpu if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Training on',DEVICE)

# Randomseed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Train variables
num_epochs = 20
learning_rate = 0.001
batch_size = 4027

# Model variables
input_size = 4
History = 3
Future = 1
num_classes = 2
K_p = 25
N_p = 1
K_q = 25
N_q = 1
num_samples = 25
hidden_history = 32
hidden_interactions = 8
hidden_future = 32
GRU_size = 128

batch_first = False
# Load data
path = pre.Path('data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)



pedestrian = pre.NodeType('pedestrian')
scene = pre.Scene(path, header=0)

x_i, x_i_fut, y_i, x_neighbour, x_i_present = scene.get_batches()

x_i = x_i.view(History, batch_size, input_size).to(torch.float32)
x_i_fut = x_i_fut.view(1, batch_size, input_size).to(torch.float32)
y_i = y_i.view(1, batch_size, 2).to(torch.float32)
x_neighbour =  x_neighbour.view(History, batch_size, input_size).to(torch.float32)

print("Loaded the data")
print(x_i.dtype)
print(x_i_fut.dtype)
print(y_i.dtype)
print(x_neighbour.dtype)
print()
print(x_i.shape)
print(x_i_fut.shape)
print(y_i.shape)
print(x_neighbour.shape)
# For debugging the forward function and model
# initialize model object
net = model(input_size, History, Future, hidden_history, hidden_interactions, hidden_future, GRU_size, batch_first, batch_size, K_p, N_p, K_q, N_q, num_samples)

# do forward function
y_true = y_i[:, :, :2]
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):     
    y_pred, M_p_norm, M_q_norm = net(x_i, x_neighbour, x_i_fut, y_i)
    optimizer.zero_grad()
    loss = loss_function(M_q_norm, M_p_norm, y_true.view(batch_size,1,2), y_pred.view(batch_size,1,25,1,6), batch_size)
    loss.backward()
    optimizer.step()

    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))