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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Training on',DEVICE)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

batch_size = 128
learning_rate = 0.005
input_size = 32 # hidden state from lstm
hidden_size = 12 # 
labels_length = 25 # 25 in motion primitives

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MNIST('./data', transform=transform, download=True)
train_data, test_data = data.random_split(dataset, (50000,10000))

train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#helper functions
def one_hot(x, max_x):
    return torch.eye(max_x + 1)[x]


    
def plot_loss(history):
    loss, val_loss = zip(*history)
    plt.figure(figsize=(15, 9))
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

num_epochs = 1000
learning_rate = 0.01

input_size = 2
hidden_size = 32
num_layers = 1
H = 3
F = 3
num_classes = 2

class model(nn.Module):
    def __init__(self, input_size, hidden_size=20):
        super(model, self).__init__()
        input_size_with_label = input_size + labels_length
        hidden_size += labels_length
        
        """
        LSTM layer that takes two inputs, x and y, has a hidden state of 32 dimensions (aka 32 features),
        consists of one layer, and takes data as input that is not batchfirst
        output is ex, the hidden state with 32 dimensions
        """

        self.history = nn.LSTM(input_size=2, hidden_size=32,num_layers=1, batch_first=False) 

        """
        Below a placeholder LSTM for the edges for modeling the interactions between pedestrians
        This has a hidden size of 8
        What is the input size???
        """

        #self.interactions = nn.LSTM(input_size=2, hidden_size=8,num_layers=1, batch_first=False) 


        """
        Attention layer needs to be incorperated below
        """

        # ADDITIVE ATTENTION LAYER HERE

        """
        Bidirectional LSTM for the prediction of the node future that is used in the kull leibner loss function.
        Input is ground truth, which is right now x and y (later vx and vy as well)
        Output is 32 dimensional hidden state that encodes the nodes future, called ey
        """

        self.future = nn.LSTM(input_size=2, hidden_size=32,num_layers=1, batch_first=False, bidirectional=True) 

        """
        fully connected layer to predict K_p x N_p matrix for discrete distribution P
        it takes the hidden state of the LSTM layer as input. (later the concatenation of history AND interaction)
        output is flattened KxN matrix
        """
        # K and N still need to be defined (their sizes)
        self.fcP = nn.Linear(32, K_p*N_p)
        # K are columns, they still need to be forced to a one-hot encoding using a proper activation function
        """
        Two fully connected layers to predict K_q and N_q for discrete distribution Q.
        It takes as input the concatenation of ex and ey
        """

        self.fcQ = nn.Linear(64, K_q*N_q)

        """
        GRU layer for decoding
        Input is of size K_p*N_p
        """

        self.gru = nn.GRU(input_size=K_p*N_p, hidden_size=128,num_layers=1, batch_first=False) 

        """
        Gaussian mixture model (GMM) for additional regularization
        Input is hidden state of GRU, so 128 features,
        Output is ???
        """
    
    """
    Below a normalize function that needs to be used after producting the distribution matrices M_p and M_q
    """
    def normalize(self, x):
        pass

    """
    Below a integrate function that needs to be used after producting the distributions according to the GMM
    The output should be the predicted future???
    """
    def integrate(self, x):
        pass

    """
    Encoding function
    """
    def encode(self, x, labels):
        x = x.view(-1, 1*28*28) # flatten
        x = torch.cat((x, labels), 1) # concatenate with labels
        x = self.relu(self.fc1(x)) # fully connected with relu
        return self.fc21(x), self.fc22(x) # return mean and variance
    
    """
    Decoding function that takes latent variable z and produces the predicted future
    """   
    def decode(self, z, labels):
        torch.cat((z, labels), 1) # concatenate latent variable with the labels
        z = self.relu(self.fc3(z)) # fully connected layer with relu
        return torch.sigmoid(self.fc4(z)) # fully connected layer with sigmoid

    """
    Reparameterize function that produces latent variable z based on matrix M_p, using N and K (not mu and variance)
    """    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 *logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    """
    Forward function, this applies the encode, reparameterize and decoding functions
    Predicts future based on (batch of) data
    """   
    def forward(self,x, labels):
        #targets = one_hot(targets,labels_length-1).float().to(DEVICE)
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar) # sample latent variable
        x = self.decode(z, labels) # 
        return x, mu, logvar

def train_cvae(net, dataloader, test_dataloader, flatten=True, epochs=20):
    validation_losses = []
    optim = torch.optim.Adam(net.parameters())

    log_template = "\nEpoch {ep:03d} val_loss {v_loss:0.4f}"
    with tqdm(desc="epoch", total=epochs) as pbar_outer:  
        for i in range(epochs):
            for batch, labels in dataloader:
                batch = batch.to(DEVICE)
                labels = one_hot(labels,9).to(DEVICE)

                if flatten:
                    batch = batch.view(batch.size(0), 28*28)

                optim.zero_grad()
                x,mu,logvar = net(batch, labels)
                loss = vae_loss_fn(batch, x[:, :784], mu, logvar)
                loss.backward()
                optim.step()
            evaluate(validation_losses, net, test_dataloader, flatten=True)
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=i+1, v_loss=validation_losses[i]))
    return validation_losses

cvae = CVAE(28*28).to(DEVICE)

def vae_loss_fn(x, recon_x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def evaluate(losses, autoencoder, dataloader, flatten=True):
    model = lambda x, y: autoencoder(x, y)[0]    
    loss_sum = []
    loss_fn = nn.MSELoss()
    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE)
        labels = one_hot(labels,9).to(DEVICE)

        if flatten:
            inputs = inputs.view(inputs.size(0), 28*28)

        outputs = model(inputs, labels)
        loss = loss_fn(inputs, outputs)            
        loss_sum.append(loss)  

    losses.append((sum(loss_sum)/len(loss_sum)).item())

history = train_cvae(cvae, train_dataset, val_dataset)

val_loss = history
plt.figure(figsize=(15, 9))
plt.plot(val_loss, label="val_loss")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()