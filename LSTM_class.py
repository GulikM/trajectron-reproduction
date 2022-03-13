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
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        out = []

        data = x
        
        # try using only 2 (x,y) hiddem dimensions; outputs
        # make sure c0 and h0 keep the right shape:
        h_0 = torch.reshape(h_0, (self.num_layers, x.size(0), self.hidden_size))
        c_0 = torch.reshape(c_0, (self.num_layers, x.size(0), self.hidden_size))
        
        y, (h_out, c_out) = self.lstm(data, (h_0, c_0))  
        
        out = y

            
            # Make the new sequence to predict the next time step
            # data = torch.FloatTensor(data[-2, -1, output])

        # # Propagate input through LSTM
        # for i in range(F):
            
        #     # make sure c0 and h0 keep the right shape:
        #     h_0 = torch.reshape(h_0, (self.num_layers, x.size(0), self.hidden_size))
        #     c_0 = torch.reshape(c_0, (self.num_layers, x.size(0), self.hidden_size))
            
        #     ula, (h_out, c_out) = self.lstm(data, (h_0, c_0))  

        #     print(ula.shape)
        #     h_out = h_out.view(-1, self.hidden_size)
            
        #     # Adjust the states (short term and long term) to predict the next time step
        #     h_0 = h_out
        #     c_0 = c_out

        #     # Predict time step and append predicted time step to the sequence
        #     output = self.fc(h_out)
        #     print(output.shape)
        #     out.append(output)
            
        #     # Make the new sequence to predict the next time step
        #     # data = torch.FloatTensor(data[-2, -1, output])
# 
        # out = torch.stack(out)
        return out


inpath = pathlib.Path('data/pedestrians/eth/train/biwi_hotel_train.txt', safe=False)

df = import_ped_data(inpath)

X, Y = get_batches(df, H=3, F=3)

X = X.type(torch.FloatTensor)
Y = Y.type(torch.FloatTensor)



num_epochs = 100
learning_rate = 0.01

input_size = 2
hidden_size = 2
num_layers = 1
H = 3
F = 3
num_classes = 2

lstm = LSTM(num_classes, input_size, hidden_size, num_layers, H, F)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
losses = []
for epoch in range(num_epochs):
    outputs = lstm(X)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, Y)
    loss.backward()   
    optimizer.step()
    losses.append(loss.item())
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# Evaluate on entire data set for comparison train and test
lstm.eval()
train_predict = lstm(X)

data_predict = train_predict.data.numpy()
data_true = Y.data.numpy()

# let's visualize one prediction step of one state:
state = 0 # x
time  = 0 # prediction for t1

plt.figure
plt.plot(data_true[time, :, state], 'r', label = 'true state_{} at future_step{}'.format(state,time))
plt.plot(data_predict[time, :, state], 'r--', label = 'prediction state_{} at future_step{}'.format(state,time))
plt.suptitle('train predictions')
plt.xlabel('batch')
plt.ylabel('state [m]')
plt.show()
plt.legend()

# plot loss
plt.figure
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('training loss')