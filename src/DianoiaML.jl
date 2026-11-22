module DianoiaML
    using BFloat16s: BFloat16
    export BFloat16
    include("tools/precision.jl")
    include("tools/precision_io.jl")
    include("tools/glorotuniform.jl")
    include("optimizer/parameter_update.jl")
    include("layer/operations.jl")
    include("layer/spatial_convolution.jl")

    struct LogitGradient{A}
        data::A
    end

    cotangent_as(prediction, delta::LogitGradient) = LogitGradient(cotangent_as(prediction, delta.data))

    activation_gradient!(dest, activation, values, delta) =
        activation.get_∇biases!(dest, values, delta)
    activation_gradient!(dest, activation, values, delta::LogitGradient) =
        copyto!(dest, delta.data)

    include("./activation_function/relu.jl")
    using .ReLU
    export ReLU
    include("./activation_function/sigmoid.jl")
    using .Sigmoid
    export Sigmoid
    include("./activation_function/softmax.jl")
    using .Softmax
    export Softmax
    include("./activation_function/tanh.jl")
    using .tanH
    export tanH
    include("./activation_function/none.jl")
    using .None
    export None
    include("activation_function/gelu.jl")
    using .GELU
    export GELU

    include("./layer/dense.jl")
    using .DenseM:Dense
    export Dense
    include("./layer/conv2d.jl")
    using .Conv2DM:Conv2D
    export Conv2D
    include("./layer/maxpooling2d.jl")
    using .MaxPooling2DM:MaxPooling2D
    export MaxPooling2D
    include("./layer/upsampling2d.jl")
    using .UpSampling2DM:UpSampling2D
    export UpSampling2D
    include("./layer/flatten.jl")
    using .FlattenM:Flatten
    export Flatten
    include("./layer/constructive.jl")
    using .ConstructiveM:Constructive
    export Constructive
    include("./layer/dropout.jl")
    using .DropoutM:Dropout
    export Dropout
    include("./layer/tconv2d.jl")
    using .TConv2DM:TConv2D
    export TConv2D
    include("network/layer_stack.jl")
    include("layer/layer_norm.jl")
    using .LayerNormM: LayerNorm
    include("layer/attention.jl")
    using .AttentionM: Attention
    export LayerNorm, Attention
    include("./network/resnet.jl")
    using .resnet:ResNet
    export ResNet
    include("./layer/residual.jl")
    using .ResidualM:Residual
    export Residual
    accepts_precision(constructor::Type{<:Union{Dense,Conv2D,MaxPooling2D,UpSampling2D,Flatten,
                                               Constructive,Dropout,TConv2D,Residual,
                                               LayerNorm,Attention}}) = true

    include("./network/sequential.jl")
    using .sequential:Sequential
    export Sequential, Hidden_Output_Layer, add_layer!
    include("./network/gan.jl")
    using .gan:GAN
    export GAN
    include("network/graph.jl")
    export Graph, Node, Add, Select, share_parameters
    include("network/transformer.jl")
    export TransformerBlock
    export AbstractLayer, initialize!, activate!, backward!, supports_backward, parameter_gradients, persistent_buffers

    include("./loss_function/categorical_cross_entropy_loss.jl")
    using .Categorical_Cross_Entropy_Loss
    export Categorical_Cross_Entropy_Loss
    include("./loss_function/binary_cross_entropy_loss.jl")
    using .Binary_Cross_Entropy_Loss
    export Binary_Cross_Entropy_Loss
    include("./loss_function/quadratic_loss.jl")
    using .Quadratic_Loss
    export Quadratic_Loss
    include("./loss_function/mean_squared_error.jl")
    using .Mean_Squared_Error
    export Mean_Squared_Error
    include("./loss_function/absolute_loss.jl")
    using .Absolute_Loss
    export Absolute_Loss
    include("loss_function/output_gradient.jl")

    include("./monitor/absolute.jl")
    using .Absolute
    export Absolute
    include("./monitor/classification.jl")
    using .Classification
    export Classification

    include("optimizer/training.jl")
    include("optimizer/training_state.jl")
    export Optimizer, TrainingState, forward!, step!, fit!, parameters, gradients, set_group!, zero_grad!
    include("./optimizer/sgd.jl")
    using .SGD
    export SGD
    include("./optimizer/minibatch_gd.jl")
    using .Minibatch_GD
    export Minibatch_GD
    include("./optimizer/adam.jl")
    using .Adam
    export Adam
    include("./optimizer/adabelief.jl")
    using .AdaBelief
    export AdaBelief
    include("./optimizer/ga.jl")
    using .GA
    export GA

    include("./tools/model_management.jl")
    include("tools/checkpoint.jl")
    export state_dict, load_state!, save_checkpoint, load_checkpoint!
    export save_Sequential, load_Sequential
    export save_GAN, load_GAN
    include("./tools/oneHot.jl")
    export oneHot

    include("precompile_workload.jl")
end
