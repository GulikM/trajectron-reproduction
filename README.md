# Reproducing Trajectron++

On this repository we have attempted to in part recreate the network proposed in [paper](https://arxiv.org/pdf/2001.03093.pdf). In particular, we sought to evaluate it from purely its textual form and disregard the existing code the authors have created. This was done for two reasons: first, this allowed for a much deeper learning experience for ourselves. Secondly, this provides a highly valuable perspective; in the pragmatic and empirical world of deep learning, papers can at times feel disjointed from their material, and come across as formal translations of a code base rather than as its conceptual foundation. Gradually, we came to the conclusion that completely ignoring the author’s code was not feasible, however, since crucial information for the implementation (voorbeeld) of the network was left out of the paper. Meaningful results were not obtained. We hypothesize that …………..

## Introduction
In the interest of road safety, reasoning about future road-user trajectories is important. *Trajectron++* was proposed as a means to that end by Salzmann et al. The network forecasts the trajectories of an arbitrary set of nodes (traffic participants), taking into consideration the node histories (past trajectories) and (optionally) some additional information such as local HD maps, raw LIDAR data, camera images, and future ego-agent motion plans. For a deep dive into how this information is turned into predictions by the network, see Sections 3 and 4 in [paper](https://arxiv.org/pdf/2001.03093.pdf).

## Results
Below one can see a plot of the training loss. As can be seen, the model learns almost nothing as the loss does not significantly decrease. This could have a variety of reasons. First among these, a mistake may have occurred in converting the complex loss function from its mathematical denotation to our coded implementation. There are for example some conversions from log probabilities to probabilities (and the other way round), which could lead to a vanishing gradient, which prevents the network from learning anything. The reason for using log probabilities is explicity to avoid this problem: log probabilities are numerically more stable, as very small probabilities are converted to large negative numbers instead of values very close to zero. Our hypothesis is that one of these conversions is implemented wrongly, causing the gradient to vanish.

  ![trainloss](/figures/trainloss.png)
  
Below a visualization of some random predicted trajectories. At first sight the result looks very promising. The plots are deceiving, however. Because of time constraints we only got the model to work for a prediction horizon of 1 timestep. This means that if we predict a velocity of 0 at every time step, the prediction of the network for the next time step will be the same as the current timestep. But since we only have a prediction horizon of 1, the current true position is updated at every prediction, resulting in a noisy prediction which lags behind. So even though the plots look visually pleasing and successful, the learned model is of no use in any practical application. 

  ![Figure3](/figures/Figure_3.png)
  ![Figure9](/figures/Figure_9.png)
  ![Figure10](/figures/Figure_10.png)
  ![Figure11](/figures/Figure_11.png)


