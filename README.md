# DianoiaML.jl

DianoiaML is a Julia library for building and training neural networks on the CPU.
It supports handwritten backpropagation, composable networks, and Transformer blocks.

## Quick start

Use Julia 1.13 or newer.

```julia
import Pkg
Pkg.add("DianoiaML")
```

```julia
using DianoiaML

model = Sequential(Dense(input_shape=(1,), layer_size=1, activation_function=None))
inputs = Float32[0 1 2]
state = TrainingState(model, Optimizer(:SGD; α=0.1))
fit!(state, [(inputs, 2f0 .* inputs)]; loss_function=Mean_Squared_Error, epochs=20)
forward!(state, Float32[3;;]; training=false)
```

[Wiki](https://github.com/SkyWorld117/DianoiaML.jl/wiki) · [Development setup](https://github.com/SkyWorld117/DianoiaML.jl/wiki/Installation#development) ·
[MNIST example](https://github.com/SkyWorld117/DianoiaML.jl/wiki/MNIST) · [Release notes](UPDATES.md)
