module Yisy_AI_Framework
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

    include("./layer/dense.jl")
    using .dense:Dense
    export Dense

    include("./network/sequential.jl")
    using .sequential:Sequential, Hidden_Output_Layer
    export Sequential, Hidden_Output_Layer

    include("./loss_function/cross_entropy_loss.jl")
    using .Cross_Entropy_Loss
    export Cross_Entropy_Loss
    include("./loss_function/quadratic_loss.jl")
    using .Quadratic_Loss
    export Quadratic_Loss
    include("./loss_function/mean_squared_error.jl")
    using .Mean_Squared_Error
    export Mean_Squared_Error
    include("./loss_function/absolute_loss.jl")
    using .Absolute_Loss
    export Absolute_Loss

    include("./optimizer/minibatch_gradient_descent.jl")
    using .Minibatch_Gradient_Descent
    export Minibatch_Gradient_Descent

    include("./tools/model_management.jl")
    export save_sequential
    export load_sequential
end
