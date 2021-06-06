# Update Log

### Update 0.4.7 - 06.06.2021
- Fixed the UI of **GA**. 
- Greatly optimized **GA** (about four times faster). 

### Update 0.4.6 - 06.06.2021
- Fixed **Minibatch_GD** and **GA**. 
- Slightly optimized **GA**. 

### Update 0.4.5 - 06.05.2021
- Added **ResNet** as a network type. 
- Added **Residual** as a layer to load **ResNet**. 

### Update 0.4.4 - 06.02.2021
- Added **TConv2D** as a layer. 

### Update 0.4.3 - 05.31.2021
- Added **Dropout** as a layer. 

### Update 0.4.2 - 05.25.2021
- Updated dependencies that solves the problem of **MaxPooling**. 

### Update 0.4.1 - 05.24.2021
- Updated dependencies. 
- Improved the speed of **MaxPooling2D** before it hangs :(. 

### Update 0.4.0 - 05.14.2021
- Rewrote the structure for better performence and more logical principles. 
- Added **Flatten** and **Constructive** as layers to reform data shape. 
- Updated **model management**. 
- Rewrote UI and simplified it. 
- Renamed to **DianoiaML.jl**. 

### Update 0.3.10 - 03.28.2021
- Support **Genetic Algorithm** officially. 
- **BLAS** is now running with a single thread in default. 
- Fixed a few bugs. 

### Update 0.3.9 - 03.15.2021
- Updated the API and it can now auto-complete most of the parameters. 
- Added **padding** and **biases** in **Conv2D**. 
- Added **UpSampling2D** as a layer. 
- Added **Genetic Algorithm** as an experimental optimizer. 
- Fixed a few bugs. 

### Update 0.3.8 - 02.24.2021
- Fixed that the speed of convergence for training convolutional networks is slower as expected. (Except with **Minibatch_GD**, I suppose this optimizer is just inefficient in this case. )
- Support **HDF5.jl** 0.15.4

### Update 0.3.7 - 02.22.2021
- Added a limit of interval **(-3.0f38, 3.0f38)** for value to prevent overflow and **NaN**. 
- **GAN** can now display the loss of discriminator. 
- Sightly improved the sytax. 

### Update 0.3.6 - 02.20.2021
- Added **GAN** as a new type of networks. 
- Split **Cross_Entropy_Loss** into **Categorical_Cross_Entropy_Loss** and **Binary_Cross_Entropy_Loss**. 
- Fixed the loss display of **AdaBelief**. 
- Known issues: there is a possibility to produce **NaN**, I am still working on it. For now, reduce the usage of **ReLU** in relatively deep networks may solve the problem. 

### Update 0.3.5 - 02.19.2021
- Fixed that **AdaBelief** activates **Adam**. 
- Optimized the structure for development of **GAN** model in the future. 
- Updated the argument keywords for `fit` function. 

### Update 0.3.4 - 02.14.2021
- Greatly improved the training speed by optimizing the structure. 
- Fixed that the filters in **Conv2D** cannot be updated until saved. 
- Fixed that the model cannot by trained multiple times. 

### Update 0.3.2 - 02.12.2021
- Added **SGD** as an optimizer.
- Optimized the structure and sytax, the "minibatch" problem is now solved. 
- Accelerated the framework by using [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl).
- Use GlorotUniform to generate random weights and biases. 

### Update 0.2.6 - 02.06.2021
- Added **Monitor** to show the current loss
- Known issues: I find out that all my optimizers update once after a batch, that means they work just like **Minibatch Gradient Descent**, so **Adam** and **AdaBelief** are not working properly but like **Minibatch Adam** and **Minibatch AdaBelief**. This slows down the training process. I will try to reconstruct the whole program in the next update. 

### Update 0.2.5 - 02.02.2021
- Greatly improved the training speed.
- In the example, it is about 20 seconds slower than Keras (epochs=5, batch_size=128). 

### Update 0.2.4 - 01.28.2021
- Added **Convolutional2D** and **MaxPooling2D** as layers.
- Added **Mean Squared Loss** as a loss function.
- Added **Adam** and **AdaBelief** as optimizers.
- Added **One Hot** and **Flatten** as tools.
- Improved the structures.
- Known issues: Convolutional2D requires a lot of RAM and is relatively slow. 