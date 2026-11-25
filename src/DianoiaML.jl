module DianoiaML
    import AcceleratedKernels as AK
    import KernelAbstractions as KA
    using KernelAbstractions: @kernel, @index, @localmem, @synchronize
    using BFloat16s: BFloat16
    using LinearAlgebra: Transpose, mul!
    using Random: Xoshiro, default_rng, rand!, shuffle!

    export BFloat16, AbstractLayer, AbstractActivation, AbstractLoss, AbstractOptimizer
    export Dense, Conv2D, TConv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
    export LayerNorm, Attention, TransformerBlock, Sequential, Residual, Graph, Node, Add, Select
    export BatchNorm2D, GlobalMeanPool2D, Embedding, PositionalEmbedding, Concatenate, Permute, Subtract
    export ResNet18, UNet, GPT, NBEATS
    export Identity, ReLU, Sigmoid, Softmax, Tanh, GELU
    export MeanSquaredError, QuadraticLoss, AbsoluteLoss, CategoricalCrossEntropy, BinaryCrossEntropy
    export CrossEntropyWithLogits, BinaryCrossEntropyWithLogits, loss_value, loss_gradient!
    export IndexedCrossEntropyWithLogits
    export SGD, Adam, AdamW, RMSProp, AdaBelief, TrainingState, batches, fit!, forward!, backward!, step!, predict
    export prepare
    export shape, parameters, gradients, set_group!, zero_grad!, persistent_buffers
    export layer_cache, layer_parameters, layer_gradients, output_shape
    export GAN, GANState, generate, fit_population!
    export Absolute, Classification, oneHot
    export state_dict, load_state!, save_checkpoint, load_checkpoint!
    export gpu_backend, to_device

    include("device.jl")
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
    include("layer/batch_norm.jl")
    include("layer/global_pooling.jl")
    include("layer/embedding.jl")
    include("layer/attention.jl")

    include("network/sequential.jl")
    include("layer/residual.jl")
    include("network/graph.jl")
    include("layer/composition.jl")
    include("network/transformer.jl")
    include("network/builtins.jl")

    include("loss_function/losses.jl")
    include("loss_function/indexed_crossentropy.jl")
    include("monitor/absolute.jl")
    include("monitor/classification.jl")
    include("optimizer/parameter_update.jl")
    include("optimizer/training.jl")
    include("optimizer/training_state.jl")
    include("optimizer/execution_plan.jl")
    include("network/gan.jl")
    include("optimizer/ga.jl")

    include("tools/checkpoint.jl")
    include("tools/oneHot.jl")
    include("gpu/kernels.jl")
    include("gpu/spatial.jl")
    include("gpu/attention.jl")
    include("gpu/training.jl")
    include("gpu/model.jl")
    include("precompile_workload.jl")
end
