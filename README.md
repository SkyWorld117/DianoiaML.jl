# DianoiaML

DianoiaML is an experimental Keras-like deep learning framework. 

The user guide and the To-Do list can be found [here](https://github.com/SkyWorld117/YisyAIFramework.jl/wiki). 

## Setup

Use Julia 1.10 or newer. The latest stable release tested here is Julia 1.13.4.
Run these commands from the repository root.

```sh
./scripts/init_env.sh
JULIA_DEPOT_PATH="$PWD/.julia-env/depot:" OPENBLAS_NUM_THREADS=1 \
  julia --project=.julia-env --threads=8 -e 'using Pkg; Pkg.test("DianoiaML")'
```

The setup script develops the local checkout and installs its dependencies in the ignored
`.julia-env/` directory. Open Julia with the same depot and project to use the package.

## Training

Supply `Float32` arrays with examples in the last dimension. Each epoch visits every
example, including the final partial batch. Use `SGD.fit` with `batch=1` for
stochastic updates, a larger batch for minibatches, or
`batch=size(input_data, ndims(input_data))` for full-batch gradient descent.
Use `shuffle=false` for ordered batches or pass an `rng` to control shuffling.
Legacy calls to `Minibatch_GD.fit` delegate to `SGD.fit`. `GA.fit` accepts
`mini_batch` as a batch-size override.

Each call to Adam or AdaBelief starts fresh optimizer moments and keeps the model's
learned parameters. To run inference, initialize the model with the number of examples
you are passing to `model.activate`. Inference disables dropout by default.

GAN training uses the discriminator's real and fake targets from the network itself.
The noise generator supplies one example per call. A discriminator can have one sigmoid
output or two outputs ordered fake, real.

`TConv2D` retains the zero-insertion and padding geometry used by existing saved models.

## Nested networks

Pass configured layers or networks to `Sequential(...)`. A `Sequential` or
`ResNet` block also works with `add_layer!(model, block)` and with the GAN
generator and discriminator layer builders.

```julia
using DianoiaML

function feedforward_block(width, hidden)
    branch = Sequential()
    add_layer!(branch, Dense; input_shape=(width,), layer_size=hidden, activation_function=ReLU)
    add_layer!(branch, Dense; layer_size=width, activation_function=None)
    return Residual(branch)
end

encoder = Sequential(feedforward_block(64, 128), feedforward_block(64, 128))
model = Sequential(encoder, Dense(input_shape=(64,), layer_size=10, activation_function=Softmax_CEL))
```

Training the outer model updates the layers inside its blocks. `Residual(block)`
requires matching input and output shapes. Call the block factory for each
position because layers keep their own forward state for backpropagation.

## Features
<details>
 <summaryClick me! ></summary>
<p>
  
- Network
  - Sequential
  - GAN
  - ResNet
- Layer
  - Flatten
  - Constructive
  - Dense
  - Convolutional2D
  - MaxPooling2D
  - UpSampling2D
  - Transposed Convolutional2D
  - Dropout
  - Residual
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
  - SGD with stochastic, minibatch, and full-batch updates
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

Install `MLDatasets` in the active environment with `import Pkg; Pkg.add("MLDatasets")`.

```julia
using MLDatasets, DianoiaML

train_x, train_y = MNIST(split=:train)[:]
dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)

model = Sequential()
model.add_layer(model, Conv2D; filter=32, input_shape=(28,28,1), kernel_size=(3,3), activation_function=ReLU)
model.add_layer(model, MaxPooling2D; pool_size=(2,2))
model.add_layer(model, Conv2D; filter=64, kernel_size=(3,3), activation_function=ReLU)
model.add_layer(model, MaxPooling2D; pool_size=(2,2))
model.add_layer(model, Flatten;)
model.add_layer(model, Dense; layer_size=128, activation_function=ReLU)
model.add_layer(model, Dense; layer_size=10, activation_function=Softmax_CEL)

Adam.fit(model=model, input_data=Float32.(reshape(train_x, 28, 28, 1, :)), output_data=oneHot(train_y, 10, dict),
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
model.add_Dlayer(model, MaxPooling2D; pool_size=(2,2))
model.add_Dlayer(model, Flatten;)
model.add_Dlayer(model, Dense; layer_size=128, activation_function=ReLU)
model.add_Dlayer(model, Dense; layer_size=64, activation_function=ReLU)
model.add_Dlayer(model, Dense; layer_size=2, activation_function=Sigmoid)

SGD.fit(model=model, input_data=Array{Float32}(reshape(train_x, 28,28,1,60000)), output_data=oneHot(train_y, 10, dict),
        loss_function=Binary_Cross_Entropy_Loss, monitor=Classification, epochs=50, batch=128)
```
  
</p>
</details>

**An example of ResNet**:
<details>
 <summaryClick me! ></summary>
<p>

```julia
using MLDatasets, DianoiaML

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()
dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)


resnet = ResNet()
resnet.add_layer(resnet, Dense; input_shape=(128,), layer_size=64, activation_function=ReLU)
resnet.add_layer(resnet, Dense; layer_size=64, activation_function=ReLU)
resnet.add_layer(resnet, Dense; layer_size=128, activation_function=ReLU)

model = Sequential()
model.add_layer(model, Conv2D; filter=32, input_shape=(28,28,1), kernel_size=(3,3), activation_function=ReLU)
model.add_layer(model, MaxPooling2D; pool_size=(2,2))
model.add_layer(model, Conv2D; filter=64, kernel_size=(3,3), activation_function=ReLU)
model.add_layer(model, MaxPooling2D; pool_size=(2,2))
model.add_layer(model, Flatten;)
model.add_layer(model, Dropout; rate=0.5)
model.add_layer(model, Dense; layer_size=128, activation_function=ReLU)
model.add_layer(model, Residual; resnet=resnet)
model.add_layer(model, Dense; layer_size=10, activation_function=Softmax_CEL)

Adam.fit(model=model, input_data=Array{Float32}(reshape(train_x, 28,28,1,60000)), output_data=oneHot(train_y, 10, dict),
        loss_function=Categorical_Cross_Entropy_Loss, monitor=Classification, epochs=10, batch=128)
```
  
</p>
</details>

Please feel free to leave comments, trouble-shootings or advice (which are very valuable for me). 
