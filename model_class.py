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
import random


class model(nn.Module):
    def __init__(self, 
                 batch_size,
                 input_size = 4, 
                 History = 3, 
                 Future = 3, 
                 hidden_history = 32, 
                 hidden_interactions = 8, 
                 hidden_future = 32, 
                 GRU_size = 128, 
                 batch_first = False, 
                 K_p = 25, 
                 N_p = 1, 
                 K_q = 25, 
                 N_q = 1,
                 num_samples = 25):
        
            super(model, self).__init__()


            # initialize parameters for the different layers
            self.input_size = input_size
            self.H = History
            self.F = Future
            self.hidden_history = hidden_history
            self.hidden_interactions = hidden_interactions
            self.hidden_future = hidden_future
            self.batch_first = batch_first
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

            self.debug = False
            self.training = True
            self.batch_size = batch_size
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

            self.h_gru_good = nn.Linear(self.K_q**self.N_q*self.num_samples,
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

        # # Below functions that are used in between layers
        # """
        # Below a normalize function that needs to be used after producting the distribution matrices M_p and M_q
        # """
    def normalize(self, M_flat, N, K):
        M = M_flat.view(-1, N, K)
        M_exp = torch.exp(M)
        if self.batch_first == False:
            row_sums = M_exp.sum(axis=0)
        else:
            row_sums = M_exp.sum(axis=1)

        M_normalized_exp = M_exp / row_sums[:, np.newaxis]
        M_normalized = torch.log(M_normalized_exp)
        return M_normalized

        # """
        # Below a integrate function that needs to be used after producting the distributions according to the GMM
        # The output should be the predicted future???
        # """
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
        z_i = self.one_hot_motion_primitives(n_samples)
        # sample every primitive ones in training, sample according to probability during inference
        if self.training:
            # output is: z_i: B x K^N x N x K
            for i in range(length):
                samples.append(z_i.tolist())
        else:
            for i in range(length):
                samples.append(random.choices(self.one_hot_motion_primitives(n_samples).tolist(), 
                                              weights=prob_tensor[i][0]/np.sum(prob_tensor[i][0]), 
                                              k=n_samples))
        return samples


    # This is the function that implements all the layers with functions in between for TRAINING
    def forward(self ,x_i, x_neighbour, x_i_fut, y_i):

        # Initialize first hidden short and long term states for history lstm
        if self.batch_first:
            self.h_0_history = Variable(torch.zeros(x_i.size(dim=0), 1, self.hidden_history))
            self.c_0_history = Variable(torch.zeros(x_i.size(dim=0), 1, self.hidden_history))
        else:
            self.h_0_history = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_history))
            self.c_0_history = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_history))
        # History forward:
        _, (self.history_h_out, c_out) = self.history(x_i, (self.h_0_history, self.c_0_history))
        
        if self.debug:
            print("History states and output:")
            print(self.h_0_history.shape)
            print(self.c_0_history.shape)
            print(self.history_h_out.shape)
            print()

        # Initialize first hidden short and long term states for interactions lstm   
        if self.batch_first:
            self.h_0_interactions = Variable(torch.zeros(x_i.size(dim=0), 1, self.hidden_interactions))
            self.c_0_interactions = Variable(torch.zeros(x_i.size(dim=0), 1, self.hidden_interactions))
        else:
            self.h_0_interactions = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_interactions))
            self.c_0_interactions = Variable(torch.zeros(1, x_i.size(dim=1), self.hidden_interactions))

        # Interactions forward:
        x_interactions = torch.cat((x_i, x_neighbour), 2) # concatenate over the features
        _, (self.interactions_h_out, c_out) = self.interactions(x_interactions, (self.h_0_interactions, self.h_0_interactions))

        if self.debug:
            print("Interactions states and output:")
            print(self.h_0_interactions.shape)
            print(self.c_0_interactions.shape)
            print(self.interactions_h_out.shape)
            print()

            
        # Initialize first hidden short and long term states for future lstm
        if self.batch_first:
            if self.training:
                self.fut_states = self.hidden_states_fut(x_i_fut.view)
        else:
            if self.training:
                self.fut_states = self.hidden_states_fut(x_i_fut.view(-1, self.input_size))
        
        if self.training:
            self.h_0_future = self.fut_states[:, 0:self.hidden_future*2]
            self.c_0_future = self.fut_states[:, 2*self.hidden_future::]
            
            self.h_0_future = self.h_0_future.reshape(2, -1, self.hidden_future)
            self.c_0_future = self.c_0_future.reshape(2, -1, self.hidden_future)
    
            # Future forward:
            _, (self.future_h_out, c_out) = self.future(x_i_fut, (self.h_0_future, self.c_0_future))
    
            if self.debug:
                print("Future states and output:")
                print(self.fut_states.shape)
                print(self.h_0_future.shape)
                print(self.c_0_future.shape)
                print(self.future_h_out.shape)
                print()

        # Create e_x and e_y
        if self.batch_first:
            self.e_x = torch.cat((self.history_h_out, self.interactions_h_out), 2).view(-1., 1, self.hidden_history + self.hidden_interactions)
            if self.training:
                self.e_y = self.future_h_out[0, :, :].view(-1, 1, self.hidden_future)
        else:
            self.e_x = torch.cat((self.history_h_out, self.interactions_h_out), 2)
            if self.training:
                self.e_y = self.future_h_out[0, :, :].view(1, -1, self.hidden_future)


        # Create inputs that generate discrete distributions matrices M_q and M_p
        if self.training:
            self.input_M_q = torch.cat((self.e_x, self.e_y), 2)
            self.M_q = self.fcQ(self.input_M_q)
            self.M_q_norm = self.normalize(self.M_q, self.N_q, self.K_q)
            
        self.input_M_p = self.e_x

        # Create the matrices of discrete distributions Q and P
        
        self.M_p = self.fcP(self.input_M_p)

        # Normalize the matrices of each distribution
        
        self.M_p_norm = self.normalize(self.M_p, self.N_p, self.K_p)

        # Sample the latent variable z_q
        #self.z_q = self.one_hot_encode_M(self.M_q_norm)
        #self.z_q = self.z_q.type(torch.FloatTensor).view(1,100,25)

        if self.batch_first:
            self.z_q_good = torch.FloatTensor(self.prob_one_hot(self.M_p_norm.tolist(), self.num_samples)).view(1, -1,self.num_samples**2).view(-1, 1, self.num_samples**2)
        else:
            self.z_q_good = torch.FloatTensor(self.prob_one_hot(self.M_p_norm.tolist(), self.num_samples)).view(1, -1,self.num_samples**2)

        # Create first hidden state for GRU layer
        #self.h_0_GRU = Variable(self.hidden_state_GRU(self.z_q))
        #self.input_GRU = torch.cat((self.z_q, self.e_x, x_i), dim=2)

        self.h_0_GRU_good = Variable(self.h_gru_good(self.z_q_good))
        
        
        # Decode with GRU layer, outputting a tensor with 128 features
        #_, self.h_out_gru = self.gru(self.input_GRU, (self.h_0_GRU))

        self.y_preds = []

        for i in range(self.num_samples):
            self.input_GRU_good = torch.cat((self.z_q_good[:,:,self.num_samples*i:self.num_samples*(i+1)], self.e_x, x_i[-1, :, :].view(1, -1, self.input_size)), dim=2)
            _, self.h_out_gru = self.gru_good(self.input_GRU_good, (self.h_0_GRU_good))

            # GMM model below, outputting the means, log_sigmas, correlation and log_probabilities
            self.mus = self.fc_mus(self.h_out_gru)
            self.log_prob = self.fc_log_prob(self.h_out_gru)
            self.log_sigmas = self.fc_log_sigmas(self.h_out_gru)
            self.corrs = self.fc_corrs(self.h_out_gru)

            
            # Integrate outputs of the GMM model
            self.mus_pos = self.integrate_mu(x_i[-1,:,:2], self.mus, self.dt)
            self.sigmas_pos = self.integrate_sigma(torch.zeros(1, self.batch_size, 2), torch.exp(self.log_sigmas), self.dt)

            self.y_pred = torch.cat((self.log_prob, self.mus_pos, self.sigmas_pos, self.corrs), dim=2)
            self.y_preds.append(self.y_pred)

        self.y_preds = torch.stack(self.y_preds)
        
        if self.training==False:
            self.y_preds = torch.mean(self.y_preds, 0).view(1, 1, -1, 6) # take mean over 25 samples during inference

        return self.y_preds, self.M_p_norm, self.M_q_norm

    def loss_function(self,M_qs, M_ps, y_true, y_pred, beta = 1, alfa =1 ):
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
        N = self.N_p
        M = 1
        K = self.K_p
        F = self.F
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
        y_true = torch.unsqueeze(torch.unsqueeze(y_true, 3), 4).reshape(-1, F, 1, 1, 2)
        
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
      
        prob = self.biv_N_pdf(x, y, mu_x, mu_y, sig_x, sig_y, rho) # size = B x F x K^N X M
        prob = torch.squeeze(torch.tensor(prob), 3) # squeeze matrix, as M = 1 for now, so we skip the weighing
        prob = torch.clamp(prob, min = 1e-5) # probability cannot be 0; otherwhise loss will be inf
        # assert(prob.shape == (B, F, K**N))
        
        weighted_prob = p_i * prob # size: B x F x K^N
        Log_likelyhood_loss_overF = -1 * torch.log(torch.sum(weighted_prob, 2)) # sum over K^N; size = B x F
       # Log_likelyhood_loss_overF = torch.rand((1000,3)) # size = B x F
        Log_likelyhood_loss = torch.mean(Log_likelyhood_loss_overF, 1) # take mean over F; size = B
        
        # calculate trainings loss:
        # print(torch.mean(beta*D_KL), torch.mean(-alfa*I_q), torch.mean(Log_likelyhood_loss))
        loss = torch.mean(beta*D_KL -alfa*I_q + Log_likelyhood_loss)
        # loss = torch.mean(Log_likelyhood_loss)
        # if self.training:
        #     errorx = ((mu_x - x).view(-1, self.F, 25 ))
        #     errory = ((mu_y - y).view(-1, self.F, 25 ))
        #     error  = (errorx**2 + errory**2)**0.5
        #     loss = torch.mean(error,(0,1,2))
        # else:
        #     errorx = ((mu_x - x).view(-1, self.F, 1 ))
        #     errory = ((mu_y - y).view(-1, self.F, 1 ))
        #     error  = (errorx**2 + errory**2)**0.5
        #     loss = torch.mean(error,(0,1,2))
        
        return loss

    def biv_N_pdf(self,x, y, mu_x, mu_y, sig_x, sig_y, rho):
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
    
    def loss_function1(self,M_qs, M_ps, y_true, y_pred, beta = 1, alfa =1 ):
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
        N = self.N_p
        M = 1
        K = self.K_p
        F = self.F
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
        y_true = torch.unsqueeze(torch.unsqueeze(y_true, 3), 4).reshape(-1, F, 1, 1, 2)
        
        # get distribution parameters:
        x = y_true[:, :, :, :, 0]
        y = y_true[:, :, :, :, 1]
        labels = y_true[:, :, :, :, 0:1]
        weight = y_pred[:, :, :, :, 0]
        mu_x = y_pred[:, :, :, :, 1]
        mu_y = y_pred[:, :, :, :, 2]
        mu = y_pred[:, :, :, :, 1:2]
        sig_x = y_pred[:, :, :, :, 3]
        sig_y = y_pred[:, :, :, :, 4]
        sig = y_pred[:, :, :, :, 3:4]
        rho   = y_pred[:, :, :, :, 5]
        
        # calculate p_i; currently only working for N=1:
        # for N=1 p_i is simply equal to Q
        assert(N==1)
        p_i = Q # size = B x N x K. We need size = B x 1 x N^K (for multiplication with prob), so valid for N=1

        prob = self.biv_N_pdf(x, y, mu_x, mu_y, sig_x, sig_y, rho) # size = B x F x K^N X M
        prob = torch.squeeze(torch.tensor(prob), 3) # squeeze matrix, as M = 1 for now, so we skip the weighing
        prob = torch.clamp(prob, min = 1e-5) # probability cannot be 0; otherwhise loss will be inf
        assert(prob.shape == (-1, F, K**N))
        
        weighted_prob = p_i * prob # size: B x F x K^N
        Log_likelyhood_loss_overF = -1 * torch.log(torch.sum(weighted_prob, 2)) # sum over K^N; size = B x F
       # Log_likelyhood_loss_overF = torch.rand((1000,3)) # size = B x F
        Log_likelyhood_loss = torch.mean(Log_likelyhood_loss_overF, 1) # take mean over F; size = B
        
        # calculate trainings loss:
        # print(torch.mean(beta*D_KL), torch.mean(-alfa*I_q), torch.mean(Log_likelyhood_loss))
        loss = torch.mean(beta*D_KL -alfa*I_q + Log_likelyhood_loss)
        # loss = torch.mean(Log_likelyhood_loss)
        errorx = ((mu_x - x).view(-1, self.F, 25 ))
        errory = ((mu_y - y).view(-1, self.F, 25 ))
        error  = (errorx**2 + errory**2)**0.5
        loss = torch.mean(error,(0,1,2))
        
        
        return loss
        


# net = model( input_size = 4, History=3, Future=1)

# # some random data that is NOT batch first (timestep, batchsize, states)
# x_i = torch.rand(3, 100, 4)
# x_neighbour = torch.rand(3, 100, 4)
# x_i_fut = torch.rand(1, 100, 4)
# y_i = torch.rand(1, 100, 2)

# # do forward function
# y_true = y_i
# y_pred, M_p_norm, M_q_norm = net(x_i, x_neighbour, x_i_fut, y_i)
# print("loss = ", net.loss_function(M_q_norm, M_p_norm, y_true.view(100,1,2), y_pred.view(100,1,25,1,6)).item())
