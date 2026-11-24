module DianoiaML
    using BFloat16s: BFloat16
    using LinearAlgebra: mul!
    using Random: Xoshiro, default_rng, rand!, shuffle!

    export BFloat16, AbstractLayer, AbstractActivation, AbstractLoss, AbstractOptimizer
    export Dense, Conv2D, TConv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
    export LayerNorm, Attention, TransformerBlock, Sequential, Residual, Graph, Node, Add, Select
    export Identity, ReLU, Sigmoid, Softmax, Tanh, GELU
    export MeanSquaredError, QuadraticLoss, AbsoluteLoss, CategoricalCrossEntropy, BinaryCrossEntropy
    export CrossEntropyWithLogits, BinaryCrossEntropyWithLogits, loss_value, loss_gradient!
    export SGD, Adam, AdaBelief, TrainingState, batches, fit!, forward!, backward!, step!, predict
    export shape, parameters, gradients, set_group!, zero_grad!, persistent_buffers
    export layer_cache, layer_parameters, layer_gradients, output_shape
    export GAN, GANState, generate, fit_population!
    export Absolute, Classification, oneHot
    export state_dict, load_state!, save_checkpoint, load_checkpoint!

    include("tools/precision.jl")
    include("tools/precision_io.jl")
    include("tools/glorotuniform.jl")
    using .GlorotUniform: GUMatrix

    include("layer/operations.jl")
    include("activation_function/activations.jl")
    include("layer/dense.jl")
    include("layer/spatial_convolution.jl")
    include("layer/conv2d.jl")
    include("layer/maxpooling2d.jl")
    include("layer/upsampling2d.jl")
    include("layer/reshape.jl")
    include("layer/dropout.jl")
    include("layer/layer_norm.jl")
    include("layer/attention.jl")

    include("network/sequential.jl")
    include("layer/residual.jl")
    include("network/graph.jl")
    include("network/transformer.jl")

    include("loss_function/losses.jl")
    include("monitor/absolute.jl")
    include("monitor/classification.jl")
    include("optimizer/parameter_update.jl")
    include("optimizer/training.jl")
    include("optimizer/training_state.jl")
    include("network/gan.jl")
    include("optimizer/ga.jl")

    include("tools/checkpoint.jl")
    include("tools/oneHot.jl")
    include("precompile_workload.jl")
end
