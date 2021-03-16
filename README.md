# YisyAIFramework

Welcome to my AI Framework project. I am new to ML/DL and I want to learn step by step how frameworks like TensorFlow, Keras and PyTorch work, by developing this project. That also means, this framework is not supposed to have a high efficiency like other professional frameworks (although the current version is ***NOT*** significantly slower than Keras at all). All of these are just for learning and fun. 

The user guide can be found [here](https://github.com/SkyWorld117/YisyAIFramework.jl/wiki). 

If you are interested in the [history versions](https://github.com/SkyWorld117/YisyAIFramework.jl/tree/history), you can also check the [update log](https://github.com/SkyWorld117/YisyAIFramework.jl/blob/master/UPDATES.md). 

Latest stable version **0.3.9** 

Environment: Julia 1.5.3

Dependencies: 
- [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) 0.15.4
- [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) 0.11.2

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
  - Genetic Algorithm (experimental)
- Tools
  - Model Management
  - One Hot
  - Flatten

Please feel free to leave comments, trouble-shootings or advice (which are very valuable for me). 
