# DianoiaML

DianoiaML is an experimental Keras-like deep learning framework. 

The user guide and the To-Do list can be found [here](https://github.com/SkyWorld117/YisyAIFramework.jl/wiki). 

Environment: Julia 1.6.1

**Features**: 
<details>
 <summaryClick me! ></summary>
<p>
  
- Network
  - Sequential
  - GAN
- Layer
  - Flatten
  - Constructive
  - Dense
  - Convolutional2D
  - MaxPooling2D
  - UpSampling2D
  - Transposed Convolutional2D
  - Dropout
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

</p>
</details>

**An example of Sequential network**:
<details>
 <summaryClick me! ></summary>
<p>

```julia
using MLDatasets, DianoiaML

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()
dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)

model = Sequential()
model.add_layer(model, Conv2D; filter=32, input_shape=(28,28,1), kernel_size=(3,3), activation_function=ReLU)
model.add_layer(model, MaxPooling2D; pool_size=(2,2))
model.add_layer(model, Conv2D; filter=64, kernel_size=(3,3), activation_function=ReLU)
model.add_layer(model, MaxPooling2D; pool_size=(2,2))
model.add_layer(model, Flatten;)
model.add_layer(model, Dense; layer_size=128, activation_function=ReLU)
model.add_layer(model, Dense; layer_size=10, activation_function=Softmax_CEL)

Adam.fit(model=model, input_data=Array{Float32}(reshape(train_x, 28,28,1,60000)), output_data=oneHot(train_y, 10, dict),
        loss_function=Categorical_Cross_Entropy_Loss, monitor=Classification, epochs=10, batch=128)
```
  
</p>
</details>

**An example of GAN**:
<details>
 <summaryClick me! ></summary>
<p>
  
```julia
using MLDatasets, DianoiaML

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()
dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)

function noise()
    return reshape([rand(1.0f0:1.0f0:10.0f0), rand(Float32)], (2,1))
end

model = GAN(noise)

model.add_Glayer(model, Dense; input_shape=(2,), layer_size=16, activation_function=ReLU)
model.add_Glayer(model, Dense; layer_size=64, activation_function=ReLU)
model.add_Glayer(model, Constructive; shape=(8,8,1))
model.add_Glayer(model, UpSampling2D; size=(2,2), activation_function=None)
model.add_Glayer(model, Flatten;)
model.add_Glayer(model, Dense; layer_size=256, activation_function=ReLU)
model.add_Glayer(model, Dense; layer_size=784, activation_function=ReLU)
model.add_Glayer(model, Constructive; shape=(28,28,1))

model.add_Dlayer(model, Conv2D; filter=16, kernel_size=(3,3), activation_function=ReLU)
model.add_Dlayer(model, Conv2D; filter=32, kernel_size=(3,3), activation_function=ReLU)
model.add_Dlayer(model, MaxPooling2D; kernel_size=(2,2), activation_function=None)
model.add_Dlayer(model, Flatten;)
model.add_Dlayer(model, Dense; layer_size=128, activation_function=ReLU)
model.add_Dlayer(model, Dense; layer_size=64, activation_function=ReLU)
model.add_Dlayer(model, Dense; layer_size=2, activation_function=Sigmoid)

SGD.fit(model=model, input_data=Array{Float32}(reshape(train_x, 28,28,1,60000)), output_data=oneHot(train_y, 10, dict),
        loss_function=Binary_Cross_Entropy_Loss, monitor=Classification, epochs=50, batch=128)
```
  
</p>
</details>

Please feel free to leave comments, trouble-shootings or advice (which are very valuable for me). 
