# DianoiaML

DianoiaML is an experimental Keras-like deep learning framework. 

The user guide and the To-Do list can be found [here](https://github.com/SkyWorld117/YisyAIFramework.jl/wiki). 

If you are interested in the [history versions](https://github.com/SkyWorld117/YisyAIFramework.jl/tree/history), you can also check the [update log](https://github.com/SkyWorld117/YisyAIFramework.jl/blob/master/UPDATES.md). 

Latest stable version **0.4.0** 

Environment: Julia 1.6.1

Dependencies: 
- [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) 0.15.4
- [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) 0.12.18
- [CheapThreads.jl](https://github.com/JuliaSIMD/CheapThreads.jl) 0.2.3

Features: 

- Network
  - Sequential
  - GAN
- Layer
  - Flatten
  - Constructive
  - Dense
  - Convolutional2D
  - MaxPooling2D **(Not recommended to use for now due to a bug caused by unkown reasons)**
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

Please feel free to leave comments, trouble-shootings or advice (which are very valuable for me). 
