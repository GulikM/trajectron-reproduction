import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import preprocessing

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
num_epochs = 1000
learning_rate = 0.01
batch_size = 200

# Model variables
input_size = 2
History = 1
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


class model(nn.Module):
    def __init__(self, input_size, H, F, hidden_history, hidden_interactions, hidden_future, GRU_size, batch_first,  batch_size, K_p, N_p, K_q, N_q, num_samples):
        super(model, self).__init__()

        # initialize parameters for the different layers
        self.input_size = input_size
        self.H = H
        self.F = F
        self.hidden_history = hidden_history
        self.hidden_interactions = hidden_interactions
        self.hidden_future = hidden_future
        self.batch_first = batch_first
        self.batch_size = batch_size
        self.K_p = K_p
        self.N_p = N_p
        self.K_q = K_q
        self.N_q = N_q
        self.GRU_size = GRU_size
        self.dt = 1
        self.num_samples = num_samples

        # GMM model parameters
        self.mus_size = 2
        self.log_prob_size = 1
        self.log_sigmas_size = 2
        self.corrs_size = 1


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
                                           self.input_size*2*self.hidden_future) # 2 times since hidden_future size since long term and short term memory need to be initialized
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

        self.h_gru_good = nn.Linear(self.K_q**self.N_q*25,
                                    self.GRU_size)

        self.gru = nn.GRU(input_size=self.input_size + self.hidden_history + self.hidden_interactions + self.K_q**self.N_q, 
                          hidden_size=self.GRU_size,
                          num_layers=1, 
                          batch_first=self.batch_first) 
        
        self.gru_good = nn.GRU(input_size=self.input_size + self.hidden_history + self.hidden_interactions + self.K_q**self.N_q, 
                          hidden_size=self.GRU_size,
                          num_layers=1, 
                          batch_first=self.batch_first) 

        """
        Gaussian mixture model (GMM) for additional regularization
        Input is hidden state of GRU, so 128 features,
        Output is ???
        """
        self.fc_mus = nn.Linear(self.GRU_size,
                             self.mus_size)

        self.fc_log_prob = nn.Linear(self.GRU_size,
                                  self.log_prob_size)

        self.fc_log_sigmas = nn.Linear(self.GRU_size,
                                    self.log_sigmas_size)
                                        
        self.fc_corrs = nn.Linear(self.GRU_size,
                               self.corrs_size)

    # Below functions that are used in between layers
    """
    Below a normalize function that needs to be used after producting the distribution matrices M_p and M_q
    """
    def normalize(self, M_flat, N, K):
        M = M_flat.view(self.batch_size, N, K)
        M_exp = torch.exp(M)
        if self.batch_first == False:
            row_sums = M_exp.sum(axis=0)
        else:
            row_sums = M_exp.sum(axis=1)

        M_normalized_exp = M_exp / row_sums[:, np.newaxis]
        M_normalized = torch.log(M_normalized_exp)
        return M_normalized

    """
    Below a integrate function that needs to be used after producting the distributions according to the GMM
    The output should be the predicted future???
    """
    def integrate_mu(self, mu, dmu, dt):
        return mu + dmu*dt

    def integrate_sigma(self, sigma, dsigma, dt):
        return sigma + dsigma*dt**2


    def one_hot_motion_primitives(self, k: int):
        motion_primitives = torch.zeros(k, k)
        for i in range(k):
            motion_primitives[i,i] = 1
        return motion_primitives

    def prob_one_hot(self, prob_tensor, n_samples: int = 25):
        length = len(prob_tensor)
        samples = []
        for i in range(length):
            samples.append(random.choices(self.one_hot_motion_primitives(n_samples).tolist(), 
                                          weights=prob_tensor[i][0]/np.sum(prob_tensor[i][0]), 
                                          k=n_samples))
        return samples


    # This is the function that implements all the layers with functions in between for TRAINING
    def forward(self ,x_i, x_neighbour, x_i_fut, y_i):

        # Initialize first hidden short and long term states for history lstm
        if self.batch_first:
            self.h_0_history = Variable(torch.zeros(1, x_i.size(dim=0), self.hidden_history))
            self.c_0_history = Variable(torch.zeros(1, x_i.size(dim=0), self.hidden_history))
        else:
            self.h_0_history = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_history))
            self.c_0_history = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_history))
        # History forward:
        _, (self.history_h_out, c_out) = self.history(x_i, (self.h_0_history, self.c_0_history))
        


        # Initialize first hidden short and long term states for interactions lstm   
        if self.batch_first:
            self.h_0_interactions = Variable(torch.zeros(1, x_i.size(dim=0), self.hidden_interactions))
            self.c_0_interactions = Variable(torch.zeros(1, x_i.size(dim=0), self.hidden_interactions))
        else:
            self.h_0_interactions = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_interactions))
            self.c_0_interactions = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_interactions))

        # Interactions forward:
        x_interactions = torch.cat((x_i, x_neighbour), 2) # concatenate over the features
        _, (self.interactions_h_out, c_out) = self.interactions(x_interactions, (self.h_0_interactions, self.h_0_interactions))

        
        # Initialize first hidden short and long term states for future lstm
        if self.batch_first:
            self.fut_states = self.hidden_states_fut(x_i_fut).view(self.input_size, self.batch_size, 2*self.hidden_future)
        else:
            self.fut_states = self.hidden_states_fut(x_i_fut).view(self.input_size, self.batch_size, 2*self.hidden_future)

        self.h_0_future = self.fut_states[:,:,0 :self.hidden_future]
        self.c_0_future = self.fut_states[:,:,self.hidden_future::]
        
        # Future forward:
        _, (self.future_h_out, c_out) = self.future(x_i_fut, (self.h_0_future, self.c_0_future))


        # Create e_x and e_y
        if batch_first:
            self.e_x = torch.cat((self.history_h_out, self.interactions_h_out), 2).view(self.batch_size, 1, self.hidden_history + self.hidden_interactions)
            self.e_y = self.future_h_out[0, :, :].view(self.batch_size, 1, self.hidden_future)
        else:
            self.e_x = torch.cat((self.history_h_out, self.interactions_h_out), 2)
            self.e_y = self.future_h_out[0, :, :].view(1, self.batch_size, self.hidden_future)


        # Create inputs that generate discrete distributions matrices M_q and M_p
        self.input_M_q = torch.cat((self.e_x, self.e_y), 2)
        self.input_M_p = self.e_x

        # Create the matrices of discrete distributions Q and P
        self.M_q = self.fcQ(self.input_M_q)
        self.M_p = self.fcP(self.input_M_p)

        # Normalize the matrices of each distribution
        self.M_q_norm = self.normalize(self.M_q, self.N_q, self.K_q)
        self.M_p_norm = self.normalize(self.M_p, self.N_p, self.K_p)

        # Sample the latent variable z_q
        #self.z_q = self.one_hot_encode_M(self.M_q_norm)
        #self.z_q = self.z_q.type(torch.FloatTensor).view(1,100,25)

        if batch_first:
            self.z_q_good = torch.FloatTensor(self.prob_one_hot(self.M_q_norm.tolist(), self.num_samples)).view(1, self.batch_size,self.num_samples**2).view(self.batch_size, 1, self.num_samples**2)
        else:
            self.z_q_good = torch.FloatTensor(self.prob_one_hot(self.M_q_norm.tolist(), self.num_samples)).view(1, self.batch_size,self.num_samples**2)

        # Create first hidden state for GRU layer
        #self.h_0_GRU = Variable(self.hidden_state_GRU(self.z_q))
        #self.input_GRU = torch.cat((self.z_q, self.e_x, x_i), dim=2)

        self.h_0_GRU_good = Variable(self.h_gru_good(self.z_q_good))
        
        
        # Decode with GRU layer, outputting a tensor with 128 features
        #_, self.h_out_gru = self.gru(self.input_GRU, (self.h_0_GRU))

        self.y_preds = []

        for i in range(self.num_samples):
            self.input_GRU_good = torch.cat((self.z_q_good[:,:,self.num_samples*i:self.num_samples*(i+1)], self.e_x, x_i), dim=2)
            _, self.h_out_gru = self.gru_good(self.input_GRU_good, (self.h_0_GRU_good))

            # GMM model below, outputting the means, log_sigmas, correlation and log_probabilities
            self.mus = self.fc_mus(self.h_out_gru)
            self.log_prob = self.fc_log_prob(self.h_out_gru)
            self.log_sigmas = self.fc_log_sigmas(self.h_out_gru)
            self.corrs = self.fc_corrs(self.h_out_gru)

            
            # Integrate outputs of the GMM model
            self.mus_pos = self.integrate_mu(x_i, self.mus, self.dt)
            self.sigmas_pos = self.integrate_sigma(torch.zeros(1, self.batch_size, 2), torch.exp(self.log_sigmas), self.dt)

            self.y_pred = torch.cat((self.log_prob, self.mus_pos, self.sigmas_pos, self.corrs), dim=2)
            self.y_preds.append(self.y_pred)

        self.y_preds = torch.stack(self.y_preds)

        return self.y_preds, self.M_p_norm, self.M_q_norm

def loss_function(M_qs, M_ps, y_true, y_pred, beta = 1, alfa =1  ):
    """
    This function must be places within a class of the nn.module
    in order to calcualte the gradients with loss.backward()
    
    Calculates loss according to eq-4 * -1 (as we want to minimize the loss, instead of maximizing)
    
    Inputs
    -------
    M_qs are B x M_q matrices with on each row the log probability, so size(M_qs) = B x N x K
    M_ps are B x M_p matrices with on each row the log probability, so size(M_ps) = B x N x K
    y_true are the true labels of size(y_true) = B x F x output_states
    y_pred are the predicted probabilty distribution parameters of size(y_pred) =  B x F x K^N x M x 6
    alfa = constant for I_q loss
    beta = constant for D_KL loss
    
    Returns
    -------
    loss = beta*D_KL -alfa*I_q + Log_likelyhood_loss

    """
    # Load parameters locally: #TODO load from self.parameter when in class
    B = batch_size
    N = 1
    M = 1
    K = 25
    F = 1
    alfa = 1
    beta = 1
    
    # convert logprobs to probs:
    Q = torch.exp(M_qs)
    P = torch.exp(M_ps)
    
    # calculate D_KL:
    D_KL = torch.sum(Q*torch.log(Q/P), (1,2)) # sum over dimensions N and K; size = B
    
    # calculate I_q:
    P_m = torch.mean(P, 1) # take mean over dimension N
    H_1 = -1 * torch.sum(P_m * torch.log(P_m), 1) # take entropy over dimension K
    H_2 = -1 * torch.sum(P * torch.log(P), 2) # take entropy over dimension K
    H_3 = torch.mean(H_2, 1) # take mean over dimension N
    I_q = H_1 - H_3 # size = B
    
    # make sure nothing is 0, otherwise loss will be inf
    assert((P_m != 0).any())
    assert((P != 0).any())
    
    # calcualte Log_likelyhood_loss:

    # first reshape y_true, such that the shape matches y_pred:
    y_true = torch.unsqueeze(torch.unsqueeze(y_true, 3), 4).reshape(B, F, 1, 1, 2)
    
    # get distribution parameters:
    x = y_true[:, :, :, :, 0]
    y = y_true[:, :, :, :, 1]
    weight = y_pred[:, :, :, :, 0]
    mu_x = y_pred[:, :, :, :, 1]
    mu_y = y_pred[:, :, :, :, 2]
    sig_x = y_pred[:, :, :, :, 3]
    sig_y = y_pred[:, :, :, :, 4]
    rho   = y_pred[:, :, :, :, 5]
    
    # calculate p_i; currently only working for N=1:
    # for N=1 p_i is simply equal to Q
    assert(N==1)
    p_i = Q # size = B x N x K. We need size = B x 1 x N^K (for multiplication with prob), so valid for N=1
  
    prob = biv_N_pdf(x, y, mu_x, mu_y, sig_x, sig_y, rho) # size = B x F x K^N X M
    prob = torch.squeeze(torch.tensor(prob), 3) # squeeze matrix, as M = 1 for now, so we skip the weighing
    prob = torch.clamp(prob, min = 1e-5) # probability cannot be 0; otherwhise loss will be inf
    assert(prob.shape == (B, F, K**N))
    
    weighted_prob = p_i * prob # size: B x F x K^N
    Log_likelyhood_loss_overF = -1 * torch.log(torch.sum(weighted_prob, 2)) # sum over K^N; size = B x F
   # Log_likelyhood_loss_overF = torch.rand((1000,3)) # size = B x F
    Log_likelyhood_loss = torch.mean(Log_likelyhood_loss_overF, 1) # take mean over F; size = B
    
    # calculate trainings loss:
    loss = torch.mean(beta*D_KL -alfa*I_q + Log_likelyhood_loss)
    return loss

def biv_N_pdf(x, y, mu_x, mu_y, sig_x, sig_y, rho):
    # make sure parameters stay wihtin range:
    sig_x = torch.clamp(sig_x, min = 1e-5).detach().numpy()
    sig_y = torch.clamp(sig_y, min = 1e-5).detach().numpy()
    rho   = torch.clamp(rho, min = 1-1e-5, max = 1-1e-5).detach().numpy()
    mu_x = mu_x.detach().numpy()
    mu_y = mu_y.detach().numpy()
    y = y.detach().numpy()
    x = x.detach().numpy()

    
    f = 1/(2*np.pi*sig_x*sig_y*np.sqrt(1-rho**2)) * np.exp(-1/(2*(1-rho**2)) * \
         (((x-mu_x)/sig_x)**2 + ((y-mu_y)/sig_y)**2 \
        -2*rho*(x-mu_x)*(y-mu_y)/(sig_x*sig_y)))                                            
    return f
        

# For debugging the forward function and model
# initialize model object
net = model(input_size, History, Future, hidden_history, hidden_interactions, hidden_future, GRU_size, batch_first, batch_size, K_p, N_p, K_q, N_q, num_samples)

# some random data that is NOT batch first (timestep, batchsize, states)
if batch_first:
    x_i = torch.rand(batch_size, 1, input_size)
    x_neighbour = torch.rand(batch_size, 1, input_size)
    x_i_fut = torch.rand(batch_size, 1, input_size)
    y_i = torch.rand(batch_size, input_size)
else:
    x_i = torch.rand(1, batch_size, input_size)
    x_neighbour = torch.rand(1, batch_size, input_size)
    x_i_fut = torch.rand(1, batch_size, input_size)
    y_i = torch.rand(1, batch_size, input_size)

# do forward function
y_true = y_i
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):     
    y_pred, M_p_norm, M_q_norm = net(x_i, x_neighbour, x_i_fut, y_i)
    optimizer.zero_grad()
    loss = loss_function(M_q_norm, M_p_norm, y_true.view(batch_size,1,2), y_pred.view(batch_size,1,25,1,6))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))