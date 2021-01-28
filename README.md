# Yisy AI Framework



Welcome to my AI Framework project. I am new to ML/DL and I want to learn step by step how frameworks like TensorFlow, Keras and PyTorch work, by developing this project. That also means, this framework is not supposed to have a high efficiency like other professional frameworks. All of these are just for learning and fun. 



Latest stable version **0.2.4** 

Environment: Julia 1.5.3

Features: 


- Network
  - Sequential
- Layer
  - Dense
  - Convolutional2D
  - MaxPooling2D
- Activation Function
  - ReLU
  - Sigmoid
  - Softmax
  - tanh
- Loss Function
  - Quadratic Loss
  - Cross Entropy Loss
  - Mean Squared Error
- Optimizer
  - Gradient Descent
  - Adam
  - AdaBelief
- Tools
  - Model Management
  - One Hot
  - Flatten

Please feel free to leave comments, trouble-shootings or advice (which are very valuable for me). 

### Update log
**Update 0.2.4 - 01.28.2021**
- Add Convolutional2D and MaxPooling2D as layers
- Add Mean Squared Loss as a loss function
- Add Adam and AdaBelief as optimizers
- Add One Hot and Flatten as tools
- Improve the structures
- The code is now completely in Julia. 
- Known issues: Convolutional2D requires a lot of RAM and is relatively slow. 


**Update 0.1.1 - 05.12.2020**
- Add tanh as an activation function
- Add model management in tools and can save and load models
- Improve the syntax slightly
- This would be the last Python version of this framework. I am re-programming this project in Julia. 
