# Update Log

### Update 0.3.9 - 03.15.2021
- Update the API and it can now auto-complete most of the parameters. 
- Add **padding** and **biases** in **Conv2D**. 
- Add **UpSampling2D** as a layer. 
- Add **Genetic Algorithm** as an experimental optimizer. 
- Fixed a few bugs. 

### Update 0.3.8 - 02.24.2021
- Fix that the speed of convergence for training convolutional networks is slower as expected. (Except with **Minibatch_GD**, I suppose this optimizer is just inefficient in this case. )
- Support **HDF5.jl** 0.15.4

### Update 0.3.7 - 02.22.2021
- Add a limit of interval **(-3.0f38, 3.0f38)** for value to prevent overflow and **NaN**. 
- **GAN** can now display the loss of discriminator. 
- Sightly improve the sytax. 

### Update 0.3.6 - 02.20.2021
- Add **GAN** as a new type of networks. 
- Split **Cross_Entropy_Loss** to **Categorical_Cross_Entropy_Loss** and **Binary_Cross_Entropy_Loss**. 
- Fix the loss display of **AdaBelief**. 
- Known issues: there is a possibility to produce **NaN**, I am still working on it. For now, reduce the usage of **ReLU** in relatively deep networks may solve the problem. 

### Update 0.3.5 - 02.19.2021
- Fix that **AdaBelief** activates **Adam**. 
- Optimize the structure for development of **GAN** model in the future. 
- Update the argument keywords for `fit` function. 

### Update 0.3.4 - 02.14.2021
- Greatly improve the training speed by optimizing the structure. 
- Fix that the filters in **Conv2D** cannot be updated until saved. 
- Fix that the model cannot by trained multiple times. 

### Update 0.3.2 - 02.12.2021
- Add **SGD** as an optimizer.
- Optimize the structure and sytax, the "minibatch" problem is now solved. 
- Accelerate the framework by using [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl).
- Use GlorotUniform to generate random weights and biases. 

### Update 0.2.6 - 02.06.2021
- Add **Monitor** to show the current loss
- Known issues: I find out that all my optimizers update once after a batch, that means they work just like **Minibatch Gradient Descent**, so **Adam** and **AdaBelief** are not working properly but like **Minibatch Adam** and **Minibatch AdaBelief**. This slows down the training process. I will try to reconstruct the whole program in the next update. 

### Update 0.2.5 - 02.02.2021
- Greatly imporve the training speed.
- In the example, it is about 20 seconds slower than Keras (epochs=5, batch_size=128). 

### Update 0.2.4 - 01.28.2021
- Add **Convolutional2D** and **MaxPooling2D** as layers.
- Add **Mean Squared Loss** as a loss function.
- Add **Adam** and **AdaBelief** as optimizers.
- Add **One Hot** and **Flatten** as tools.
- Improve the structures.
- The code is now completely in Julia. 
- Known issues: Convolutional2D requires a lot of RAM and is relatively slow. 

### Update 0.1.1 - 05.12.2020
- Add **tanh** as an activation function.
- Add **model management** in tools and can save and load models.
- Improve the syntax slightly.
- This would be the last Python version of this framework. I am re-programming this project in Julia. 
