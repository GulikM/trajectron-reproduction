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

    def __init__(self, num_classes, input_size, hidden_size, num_layers, H, F):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.H = H
        self.F = F
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=False)
        
        self.fc = nn.Linear(hidden_size, num_classes) # deze moet eruit later om de hidden state naar de encoder te brengen

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(dim=1), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(dim=1), self.hidden_size))

        out = []


        data = x
        # Propagate input through LSTM
        for i in range(F):
            
            ula, (h_out, c_out) = self.lstm(data, (h_0, c_0))  
            h_out = h_out.view(-1, self.hidden_size)
            
            # Adjust the states (short term and long term) to predict the next time step
            h_0 = h_out
            c_0 = c_out

            # make sure c0 and h0 keep the right shape:
            h_0 = torch.reshape(h_0, (self.num_layers, data.size(1), self.hidden_size))
            c_0 = torch.reshape(c_0, (self.num_layers, data.size(1), self.hidden_size))

            # Predict time step and append predicted time step to the sequence
            output = self.fc(h_out) # dit gaat er dus uit, output moet een tuple van hidden states worden van lengte F
            out.append(output)
            
            # Make the new sequence to predict the next time step
            l =  [data[2, :, :], data[1, :, :], output]
            data = torch.stack(l)

            
        out = torch.stack(out)
        return out


inpath = pathlib.Path('data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)

df = import_ped_data(inpath)

X, Y = get_batches(df, H=3, F=3)

X = X.type(torch.FloatTensor)
Y = Y.type(torch.FloatTensor)

num_epochs = 1000
learning_rate = 0.01

input_size = 2
hidden_size = 32
num_layers = 1
H = 3
F = 3
num_classes = 2

lstm = LSTM(num_classes, input_size, hidden_size, num_layers, H, F)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(X)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, Y)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# Evaluate on entire data set for comparison train and test
lstm.eval()
train_predict = lstm(X)

data_predict = train_predict.data.numpy()
dataY_plot = Y.data.numpy()

train_size = len(X)
