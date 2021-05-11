module DianoiaML
    using LinearAlgebra
    BLAS.set_num_threads(1)

    include("./activation_function/relu.jl")
    using .ReLU
    export ReLU
    include("./activation_function/sigmoid.jl")
    using .Sigmoid
    export Sigmoid
    include("./activation_function/softmax.jl")
    using .Softmax
    export Softmax
    using .Softmax_CEL
    export Softmax_CEL
    include("./activation_function/tanh.jl")
    using .tanH
    export tanH
    include("./activation_function/none.jl")
    using .None
    export None

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

    include("./network/sequential.jl")
    using .sequential:Sequential, Hidden_Output_Layer
    export Sequential, Hidden_Output_Layer
    include("./network/gan.jl")
    using .gan:GAN
    export GAN

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

    include("./monitor/absolute.jl")
    using .Absolute
    export Absolute
    include("./monitor/classification.jl")
    using .Classification
    export Classification

    include("./optimizer/minibatch_gd.jl")
    using .Minibatch_GD
    export Minibatch_GD
    include("./optimizer/adam.jl")
    using .Adam
    export Adam
    include("./optimizer/adabelief.jl")
    using .AdaBelief
    export AdaBelief
    include("./optimizer/sgd.jl")
    using .SGD
    export SGD
    include("./optimizer/ga.jl")
    using .GA
    export GA

    include("./tools/model_management.jl")
    export save_Sequential, load_Sequential
    export save_GAN, load_GAN
    include("./tools/oneHot.jl")
    export oneHot
end