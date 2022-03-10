# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:38:02 2022

@author: maart

Beun script to try out the GNU decoder part of trajectron
"""

# import torch
# import torch.nn as nn




# rnn   = nn.GRU(H_in, H_out, num_layers = layers) # initiate GRU object structure
# input = torch.randn(L, N, H_in) # generate some random input
# h0    = torch.randn(D*layers, N, H_out)   # init weights randomly

# output, hn = rnn(input, h0) 

# Q: So training and inference at same time? Or use Hn i future?
# Q: many to one? many to many?
# Q: training for all models combined??


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 

num_par = 10 # output
num_epochs = 2
batch_size = 100
learning_rate = 0.001
input_size = 4
sequence_length = 10
hidden_size = 128
num_layers = 1

# Data 
input = torch.randn(sequence_length, batch_size, hidden_size) # generate some random input


# Fully connected neural network with one hidden layer
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_par):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.GRU = nn.GRU(input_size, hidden_size, num_layers)

        self.fc = nn.Linear(hidden_size, num_par)
        
    def forward(self, x):
        # Set initial hidden states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # Forward propagate RNN
        out, _ = self.GRU(x, h0)  
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :] # this means many to one!
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out

model = GRU(input_size, hidden_size, num_layers, num_par).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = batch_size
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')