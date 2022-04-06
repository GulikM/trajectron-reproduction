import torch
import numpy as np

def loss_function(M_qs, M_ps, y_true, y_pred, batch_size, beta = 1, alfa =1):
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
        