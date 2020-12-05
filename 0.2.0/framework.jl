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

    include("./optimizer/stochastic_gradient_descent.jl")
    using .Stochastic_Gradient_Descent
    export Stochastic_Gradient_Descent

    include("./tools/model_management.jl")
    export save_sequential
    export load_sequential
end
