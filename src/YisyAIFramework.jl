module YisyAIFramework
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
    using .dense:Dense
    export Dense
    include("./layer/conv2d.jl")
    using .conv2d:Conv2D
    export Conv2D
    include("./layer/maxpooling2d.jl")
    using .maxpooling2d:MaxPooling2D
    export MaxPooling2D
    include("./layer/upsampling2d.jl")
    using .upsampling2d:UpSampling2D
    export UpSampling2D

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
    include("./tools/one_hot.jl")
    export One_Hot
    include("./tools/flatten.jl")
    export flatten
end
