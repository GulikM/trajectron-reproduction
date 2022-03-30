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

num_epochs = 1000
learning_rate = 0.01

input_size = 2
hidden_size = 32
num_layers = 1
H = 3
F = 3
num_classes = 2
K_p = 25
N_p = 1
K_q = 25
N_q = 1

hidden_history = 32
hidden_interactions = 8
hidden_future = 32

batch_first = False

GRU_size = 128

class model(nn.Module):
    def __init__(self, input_size, H, F, hidden_history, hidden_interactions, hidden_future, GRU_size, batch_first, K_p, N_p, K_q, N_q):
        super(model, self).__init__()

        # initialize parameters for the different layers
        self.input_size = input_size
        self.H = H
        self.F = F
        self.hidden_history = hidden_history
        self.hidden_interactions = hidden_interactions
        self.hidden_future = hidden_future
        self.batch_first = batch_first
        self.K_p = K_p
        self.N_p = N_p
        self.K_q = K_q
        self.N_q = N_q
        self.GRU_size = GRU_size


        # Below the initialization of the layers used in the model
        """
        LSTM layer that takes two inputs, x and y, has a hidden state of 32 dimensions (aka 32 features),
        consists of one layer, and takes data as input that is not batchfirst
        output is ex, the hidden state with 32 dimensions
        """

        self.history = nn.LSTM(input_size=self.input_size, 
                               hidden_size=self.hidden_history,
                               num_layers=1, 
                               batch_first=self.batch_first) 

        """
        Below a placeholder LSTM for the edges for modeling the interactions between pedestrians
        This has a hidden size of 8
        What is the input size???
        """

        self.interactions = nn.LSTM(input_size=2*self.input_size, # 2 times since concatanation of node position and pedestrian interaction vector
                                    hidden_size=self.hidden_interactions,
                                    num_layers=1, 
                                    batch_first=self.batch_first) 

        """
        Bidirectional LSTM for the prediction of the node future that is used in the kull leibner loss function.
        Input is ground truth, which is right now x and y (later vx and vy as well)
        Output is 32 dimensional hidden state that encodes the nodes future, called ey
        """
        # Use linear layer once to initialize the first long term and short term states

        self.hidden_states_fut = nn.Linear(self.input_size, 
                                           2*2*self.hidden_future) # 2 times since hidden_future size since long term and short term memory need to be initialized
                                                                 # also 2 times for the bidirectional part


        self.future = nn.LSTM(input_size=self.input_size, 
                              hidden_size=self.hidden_future,
                              num_layers=1, 
                              batch_first=self.batch_first, 
                              bidirectional=True) 

        """
        fully connected layer to predict K_p x N_p matrix for discrete distribution P
        it takes the hidden state of the LSTM layer as input. (later the concatenation of history AND interaction)
        output is flattened KxN matrix
        """
        # K and N still need to be defined (their sizes)
        self.fcP = nn.Linear(self.hidden_history + self.hidden_interactions, 
                             self.K_p*self.N_p)
        # K are columns, they still need to be forced to a one-hot encoding using a proper activation function
        """
        Two fully connected layers to predict K_q and N_q for discrete distribution Q.
        It takes as input the concatenation of ex and ey
        """

        self.fcQ = nn.Linear(self.hidden_history + self.hidden_interactions + self.hidden_future, 
                             self.K_q*self.N_q)

        """
        GRU layer for decoding
        Input is of size K_p*N_p
        """
        # K_q**N_q is latent variable z_i
        # Done once to initialize the first hidden state for the GRU
        self.hidden_state_GRU = nn.Linear(self.K_q**self.N_q,
                                          self.GRU_size)


        self.gru = nn.GRU(input_size=hidden_history*hidden_interactions*K_q**N_q, 
                          hidden_size=self.GRU_size,
                          num_layers=1, 
                          batch_first=self.batch_first) 

        """
        Gaussian mixture model (GMM) for additional regularization
        Input is hidden state of GRU, so 128 features,
        Output is ???
        """
    
    # Below functions that are used in between layers
    """
    Below a normalize function that needs to be used after producting the distribution matrices M_p and M_q
    """
    def normalize(self, M_flat, N, K):
        batch_size = 100
        M = M_flat.view(batch_size, N, K)
        M_exp = torch.exp(M)
        row_sums = M_exp.sum(axis=1)
        M_normalized_exp = M_exp / row_sums[:, np.newaxis]
        M_normalized = torch.log(M_normalized_exp)
        return M_normalized

    """
    Below a integrate function that needs to be used after producting the distributions according to the GMM
    The output should be the predicted future???
    """
    def integrate(self, x):
        pass

    """
    Sample function that produces latent variable z based on matrix M_p, using N and K (not mu and variance)
    """    
    def one_hot_encode_M(self, M):
        return 
    
    # This is the function that implements all the layers with functions in between for TRAINING
    def forward(self ,x_i, x_neighbour, x_i_fut, y_i):

        # Initialize first hidden short and long term states for history lstm
        self.h_0_history = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_history))
        self.c_0_history = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_history))
        
        # History forward:
        _, (self.history_h_out, c_out) = self.history(x_i, (self.h_0_history, self.c_0_history))
        


        # Initialize first hidden short and long term states for interactions lstm   
        self.h_0_interactions = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_interactions))
        self.c_0_interactions = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_interactions))

        # Interactions forward:
        x_interactions = torch.cat((x_i, x_neighbour), 2) # concatenate over the features
        _, (self.interactions_h_out, c_out) = self.interactions(x_interactions, (self.h_0_interactions, self.h_0_interactions))

        
        # Initialize first hidden short and long term states for future lstm
        self.fut_states = self.hidden_states_fut(x_i_fut).view(2,100,64)

        
        self.h_0_future = self.fut_states[:,:,0 :self.hidden_future]
        self.c_0_future = self.fut_states[:,:,self.hidden_future::]
        
        # Future forward:
        _, (self.future_h_out, c_out) = self.future(x_i_fut, (self.h_0_future, self.c_0_future))

    
        # Create e_x and e_y
        self.e_x = torch.cat((self.history_h_out, self.interactions_h_out), 2)
        self.e_y = self.future_h_out[0, :, :].view(1, 100, 32)


        # Create inputs that generate discrete distributions matrices M_q and M_p
        self.input_M_q = torch.cat((self.e_x, self.e_y), 2)
        self.input_M_p = self.e_x

        # Create the matrices of discrete distributions Q and P
        self.M_q = self.fcQ(self.input_M_q)
        self.M_p = self.fcP(self.input_M_p)

        # Normalize the matrices of each distribution
        self.M_q_norm = self.normalize(self.M_q, self.N_q, self.K_q)
        self.M_p_norm = self.normalize(self.M_p, self.N_p, self.K_p)

        return 0

    

# For debugging the forward function and model
# initialize model object
net = model(input_size, H, F, hidden_history, hidden_interactions, hidden_future, GRU_size, batch_first, K_p, N_p, K_q, N_q)

# some random data that is NOT batch first (timestep, batchsize, states)
x_i = torch.rand(1, 100, 2)
x_neighbour = torch.rand(1, 100, 2)
x_i_fut = torch.rand(1, 100, 2)
y_i = torch.rand(1, 100, 2)

# do forward function
net.forward(x_i, x_neighbour, x_i_fut, y_i)