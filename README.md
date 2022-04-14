# Trajectron++ reproducibility project

## Results
Below one can see a plot of the training loss. As can be seen, the model learns almost nothing as the loss does not significantly decrease. This could have a variety of reasons. First among these, a mistake may have occurred in converting the complex loss function from its mathematical denotation to our coded implementation. There are for example some conversions from log probabilities to probabilities (and the other way round), which could lead to a vanishing gradient, which prevents the network from learning anything. The reason for using log probabilities is to avoid this problem: log probabilities are numerically more stable, as very small probabilities are converted to large negative numbers instead of values very close to zero. Our hypothesis is that one of these conversions is implemented wrongly, causing the gradient to vanish.

  ![trainloss](/figures/trainloss.png)
  
Below a visualization of some random predicted trajectories. At first sight the result looks very promising, these plots are however deceiving. Because of time constraints we only got the model to work for a prediction horizon of 1 timestep. This means that if we predict a velocity of 0 at every time step, our prediction for the next time step will be the same as the current timestep. But since we only have a prediction horizon of 1, the current true position is updated at every prediction, resulting in a noisy prediction which lags behind. So even though the plots look visually pleasing and successful, the learned model is of no use in any practical application. 

  ![Figure3](/figures/Figure_3.png)
  ![Figure9](/figures/Figure_9.png)
  ![Figure10](/figures/Figure_10.png)
  ![Figure11](/figures/Figure_11.png)


