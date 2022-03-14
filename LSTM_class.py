import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from preprocessing import get_batches
from preprocessing import import_ped_data
import pandas as pd
import pathlib

"""
For the future prediction based on the history of an agent the following parameters for the LSTM are used in the paper:
    Hidden dim = 32
    Input size = 2
    Num_layers = 1
    Num_classes = 2
    History length H = 3 
    Future length F = 3 
"""

class LSTM(nn.Module):

    def __init__(self, num_states, input_size, hidden_size, num_layers, H, F):
        super(LSTM, self).__init__()
        
        self.num_states = num_states
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.H = H
        self.F = F
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_states)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        data = x
        
        # make sure c0 and h0 keep the right shape:
        h_0 = torch.reshape(h_0, (self.num_layers, x.size(0), self.hidden_size))
        c_0 = torch.reshape(c_0, (self.num_layers, x.size(0), self.hidden_size))
        
        y, (h_out, c_out) = self.lstm(data, (h_0, c_0))  
        y = self.fc(y)
        
        out = y

        return out


inpath = pathlib.Path('data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)
df_train = import_ped_data(inpath)
inpath = pathlib.Path('data/pedestrians/eth/val/biwi_hotel_val.txt', safe=False)
df_val = import_ped_data(inpath)

input_size = 2
hidden_size = 32
num_layers = 1
H = 3
F = 3
num_states = 2

X_train, Y_train = get_batches(df_train, H=H, F=F)
X_val, Y_val = get_batches(df_val, H=H, F=F)



num_epochs = 100
learning_rate = 0.01

lstm = LSTM(num_states, input_size, hidden_size, num_layers, H, F)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
losses_train = []
losses_val   = []

for epoch in range(num_epochs):
    outputs = lstm(X_train)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, Y_train)
    loss.backward()   
    optimizer.step()
    losses_train.append(loss.item())
    loss_val = criterion(lstm(X_val), Y_val)
    losses_val.append(loss_val.item())
    
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# Evaluate on entire data set for comparison train and test
lstm.eval()
train_predict = lstm(X_train)

data_predict = train_predict.data.numpy()
data_true = Y_train.data.numpy()

# let's visualize one prediction step of one state:
state = 0 # x
time  = 0 # prediction for t1

# train set
plt.figure()
plt.plot(data_true[time, :, state], 'r', label = 'true state_{} at future_step{}'.format(state,time))
plt.plot(data_predict[time, :, state], 'r--', label = 'prediction state_{} at future_step{}'.format(state,time))
plt.suptitle('train predictions')
plt.xlabel('batch')
plt.ylabel('state [m]')
plt.show()
plt.legend()


val_predict  = lstm(X_val)
data_predict = val_predict.data.numpy()
data_true = Y_val.data.numpy()

# validation set
plt.figure()
plt.plot(data_true[time, :, state], 'r', label = 'true state_{} at future_step{}'.format(state,time))
plt.plot(data_predict[time, :, state], 'r--', label = 'prediction state_{} at future_step{}'.format(state,time))
plt.suptitle('validation predictions')
plt.xlabel('batch')
plt.ylabel('state [m]')
plt.show()
plt.legend()



# plot losses
plt.figure()
plt.plot(losses_train, label = 'training loss')
plt.plot(losses_val, label = 'validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()