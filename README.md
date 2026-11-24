# DianoiaML.jl

DianoiaML is a Julia library for building and training neural networks on CPUs and GPUs.
It supports handwritten backpropagation, composable networks, and Transformer blocks.

## Quick start

Use Julia 1.13 or newer.

```julia
import Pkg
Pkg.add("DianoiaML")
```

```julia
using DianoiaML

model = Sequential(Dense(1, 1))
inputs = Float32[0 1 2]
state = TrainingState(model, SGD(; lr=0.1))
fit!(state, batches(inputs, 2f0 .* inputs; batch_size=3); loss=MeanSquaredError(), epochs=20)
predict(model, Float32[3;;])
```

[Wiki](https://github.com/SkyWorld117/DianoiaML.jl/wiki) · [Development setup](https://github.com/SkyWorld117/DianoiaML.jl/wiki/Installation#development) ·
[MNIST example](https://github.com/SkyWorld117/DianoiaML.jl/wiki/MNIST) · [GPU training](https://github.com/SkyWorld117/DianoiaML.jl/wiki/GPU) · [Release notes](UPDATES.md)
