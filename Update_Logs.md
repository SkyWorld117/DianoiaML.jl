# Update Log

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
