# Reproducing Trajectron++

Maarten Hugenholtz | 4649516 | M.D.Hugenholtz@student.tudelft.nl \
Mats van der Gulik | 4651286 | M.A.vanderGulik@student.tudelft.nl \
Stijn Lafontaine   | 4908457 | S.C.Lafontaine@student.tudelft.nl \
Dirk

On this repository we have attempted to in part recreate the proposed network called [*Trajectron++*](https://arxiv.org/abs/2001.03093). In particular, we sought to evaluate it from purely its textual form and disregard the [existing code](https://github.com/StanfordASL/Trajectron-plus-plus) the authors have created. This was done for two reasons: first, this allowed for a much deeper learning experience for ourselves. Secondly, this provides a highly valuable perspective; in the pragmatic and empirical world of deep learning, papers can at times feel disjointed from their material, and come across as formal translations of a code base rather than as its conceptual foundation. Gradually, we came to the conclusion that completely ignoring the author’s code was not feasible, however, since crucial information for the implementation of the network was left out of the paper. Meaningful results were not obtained. We believe that this has to do with the vanishing gradient problem, possibly caused by a mistake made in rewriting the loss function from its mathematical denotation to functioning code.

## Introduction

In the interest of road safety, predicting future road-user trajectories is important for autonomous agents in this environment. *Trajectron++* was proposed as a means to that end by Salzmann et al. The network forecasts the trajectories of an arbitrary set of nodes (traffic participants), taking into consideration the node histories (past trajectories) and - optionally - some additional information such as local HD maps, raw LIDAR data, camera images and future ego-agent motion plans. For a deep dive into how this information is turned into predictions by the network, see Sections 3 and 4 in [the paper](https://arxiv.org/abs/2001.03093).

## Our Network Structure

Considering the restricted time available and the breadth of knowledge that needed to be acquired, it was decided to scale down from what was presented in the paper. Notably, the geography data (specifically the HD maps) was omitted as it was not directly available and seemed only supplementary to the concepts that the original paper meant to tackle. 
The author’s evaluated their *Trajectron++* framework on three publicly-available datasets: *ETH*, *UCY*, and *nuScenes*. We chose to limit our implementation to the *ETH* dataset, however. This dataset consists only of pedestrian trajectories, which, by our understanding, renders the attention layer obsolete since we need not compare different types of road users with each other.
As can be seen in the network schematic, this leaves us with a network consisting of three types of layers: multiple Long Short Term Memory networks (LSTMs), a single Gated Recurrent Unit (GRU), and multiple fully connected layers (FCs). Together, they make up different architectures, one of which is a Conditional Variational Auto-encoder (CVAE) which produces the parameters of a Gaussian Mixture Model (GMM), which is eventually used for the final trajectory predictions.

### Pre-processing

The first step to reproduce the *Trajectron++* model was to preprocess the raw data from the ETH dataset to something we can feed in the network. In the raw data, some nodes (pedestrians in the case of the ETH dataset) are very long in the area where the positions were recorded, whilst others only briefly appear. In other words, the sequence length varies a lot while we can only feed a constant sequence length into the LSTMs in the beginning of the network. To overcome this problem we used a sliding window approach, where we iterate over all nodes and all timesteps for each individual node where enough data is available. For 1 node and 1 timestep the input data and the corresponding label has the following format:
- `X_node`: History x input_states
- `Y_node`: Future x output_states

Where, ‘History’ is the number of previous timesteps (including the present) which we use to predict the next location, and ‘Future’ is the number of future timesteps we wish to predict. ‘Input_states’ is the number of states we use for the prediction which in our case is the position (`x`,`y`) and the velocity (`xdot`, `ydot`) of a particular node. For the ‘output_states’ the states only consisted of the x- and y-position. Besides `X_node` and `Y_node`, the preprocessing function also produces `X_neighbours`, which is a tensor where information of neighbour nodes (nodes within a specified perception range) is aggregated, and `X_fut`, which is used to model the q-distribution (only used during training). 
We iterate through the whole dataset and when there is enough data available (to satisfy the minimum history and future sequence length), the data is stacked together to create batches, which are eventually fed into the network. 

### LSTM

The LSTMs are used to find the features that are most important from the input sequence. The LSTMs do this by using three different gates that decide what of the previous hidden state, previous memory state and input is remembered in the memory cell and what of the previous hidden state, previous memory state and input  is pushed to the hidden state. The inner workings of recurrent networks like LSTMs and GRUs are explained very well in this [blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). Three LSTM layers are used in the network. One for producing features based on the previous positions of the pedestrian that is considered. This LSTM is called the *History LSTM*. The second LSTM is used to produce features that model the interaction between the pedestrian considered and all pedestrians within a certain range of that pedestrian. This LSTM is called the *Interaction LSTM*. The last LSTM is only used during training. This is a bidirectional LSTM that produces features based on the future of the considered pedestrian. This LSTM is called the *Future LSTM*. Only the features produced by the *History LSTM* and *Interaction LSTM* are used to generate discrete distribution P. All features produced by the three LSTMs are used to generate discrete distribution Q. How and why these distributions come into existence is further explained in the CVAE section.

### GRU

The GRU uses two gates to decide what of the previous hidden state is used and what of the input is used to create the new hidden state. This is a simpler layer than the LSTM, but due to its simplicity can perform better in certain circumstances. Again, see this [blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) for a detailed explanation. The GRU is used in the network to generate the 128 features that are used to produce the Gaussian Mixture Model of the predictions (further explained in the GMM section). It does this by taking as input the latent variable sampled from one of the distributions concatenated with the features produced by the *History LSTM* and *Interaction LSTM* and the current position of the considered pedestrian. For variable steps into the future, the prediction of the previous step is used as the current position of the considered pedestrian.

### CVAE

This is a part that consists of fully connected layers, making an encoder that produces in our case a discrete distribution, and a decoder that then turns samples back to predicted states. Two discrete distributions are learned, namely Q and P. Q is based on all features produced by the three LSTMs, which can only be known during training. P is based on features produced by the *History LSTM* and the *Interaction LSTM*.
Then 25 samples are sampled from a distribution (Q for training, P for inference). These samples are then sequentially fed to the GRU, producing 25 different outputs per timestep. These 25 outputs will then result in 25 different predictions, one for each sample from the latent variable space. The predicted value for the future time step is the mean of these 25 different predictions.

### GMM

This part consists of four fully connected layers in parallel. The goal of these layers is to produce a Gaussian Mixture Model of the hidden state output of the GRU. The outputs of these layers should then be
The logarithmic probabilities of the prediction
The mean values of the prediction (which are velocity of x and y)
The logarithmic variance of the prediction (also one for vx and one for vy
The correspondence between the predicted values (correlations between vx and vy)
After computing these values, the means and variances are then integrated to produce the means of the predicted positions x and y and the variances of the predicted positions x and y. These mean values can then be looped back into the GRU if one would want to predict another step into the future based on the already available history and interactions data. 

## Results
Below one can see a plot of the training loss. As can be seen, the model learns almost nothing as the loss does not significantly decrease. This could have a variety of reasons. First among these, a mistake may have occurred in converting the complex loss function from its mathematical denotation to our coded implementation. There are for example some conversions from log probabilities to probabilities (and the other way round), which could lead to a vanishing gradient, which prevents the network from learning anything. The reason for using log probabilities is explicity to avoid this problem: log probabilities are numerically more stable, as very small probabilities are converted to large negative numbers instead of values very close to zero. Our hypothesis is that one of these conversions is implemented wrongly, causing the gradient to vanish.

  ![trainloss](/figures/trainloss.png)
  
Below a visualization of some random predicted trajectories. At first sight the result looks very promising. The plots are deceiving, however. Because of time constraints we only got the model to work for a prediction horizon of 1 timestep. This means that if we predict a velocity of 0 at every time step, the prediction of the network for the next time step will be the same as the current timestep. But since we only have a prediction horizon of 1, the current true position is updated at every prediction, resulting in a noisy prediction which lags behind. So even though the plots look visually pleasing and successful, the learned model is of no use in any practical application. 

  ![Figure3](/figures/Figure_3.png)
  ![Figure9](/figures/Figure_9.png)
  ![Figure10](/figures/Figure_10.png)
  ![Figure11](/figures/Figure_11.png)


