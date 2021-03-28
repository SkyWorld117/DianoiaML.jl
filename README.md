# YisyAIFramework

YisyAIFramework is an experimental Keras-like deep learning framework. It should work correctly, however, please reconsider if you are preparing to use it in serious researches. 

The user guide and the To-Do list can be found [here](https://github.com/SkyWorld117/YisyAIFramework.jl/wiki). 

If you are interested in the [history versions](https://github.com/SkyWorld117/YisyAIFramework.jl/tree/history), you can also check the [update log](https://github.com/SkyWorld117/YisyAIFramework.jl/blob/master/UPDATES.md). 

Latest stable version **0.3.10** 

Environment: Julia 1.5.3

Dependencies: 
- [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) 0.15.4
- [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) 0.11.2

Features: 

- Network
  - Sequential
  - GAN
- Layer
  - Dense
  - Convolutional2D
  - MaxPooling2D
  - UpSampling2D
- Activation Function
  - ReLU
  - Sigmoid
  - Softmax
  - tanh
- Loss Function
  - Quadratic Loss
  - Categorical Cross Entropy Loss
  - Binary Cross Entropy Loss
  - Mean Squared Error
- Monitor
  - Absolute
  - Classification
- Optimizer
  - Minibatch Gradient Descent
  - Stochastic Gradient Descent
  - Adam
  - AdaBelief
  - Genetic Algorithm
- Tools
  - Model Management
  - One Hot
  - Flatten

Please feel free to leave comments, trouble-shootings or advice (which are very valuable for me). 
