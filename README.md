# Trajectron++ reproducibility project

## Results
Below one can see a plot of the training loss. As can be seen, the model learns almost nothing as the loss does not significantly decrease. This could have a variety of reasons. First among these, a mistake may have occurred in converting the complex loss function from its mathematical denotation to our coded implementation. There are for example some conversions from log probablities to probabilities (and the other way round), which could lead to a vanishing gradient, which prevents the network from learning anything. The reason for using log probabilities is to avoid this problem. Our hypothesis is that one of these conversion is implemnted wronly, causing the gradient to vanish. 

  ![trainloss](/figures/trainloss.png)
  
Below a visualization of some random predicted trajectories. At first sight the result looks very promomsing, these plots are however deceiving. Because of time constraints we only got the model to work for a prediction horizon of 1 timestep. Thus, even when the  
![Figure3](/figures/Figure_3.png)
![Figure9](/figures/Figure_9.png)
![Figure10](/figures/Figure_10.png)
![Figure11](/figures/Figure_11.png)


