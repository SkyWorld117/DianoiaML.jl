module resnet
    using LoopVectorization

    mutable struct ResNet
        layers::Array{Any}
        add_layer::Any
        activate::Any
        initialize::Any
        update::Any
        num_layer::Int64

        loss::Float64

        default_input_shape::Tuple

        function ResNet()
            new(Any[Hidden_Input_Layer(Float32[])], ResNet_add_layer, activate_ResNet, init_ResNet, update_ResNet, 0, 0.0, ())
        end
    end

    mutable struct Hidden_Input_Layer
        output::Array{Float32}
    end
    mutable struct Hidden_Output_Layer
        δ::Array{Float32}
    end

    function ResNet_add_layer(model::ResNet, layer::Any; args...)
        push!(model.layers, layer(;input_shape=model.default_input_shape, args...))
        model.num_layer += 1
        model.default_input_shape = model.layers[end].output_shape
    end

    function init_ResNet(model::ResNet, mini_batch::Int64)
        if length(model.layers)==model.num_layer+1
            push!(model.layers, Hidden_Output_Layer(zeros(Float32, model.layers[end].output_shape..., mini_batch)))
        end
        for i in 2:length(model.layers)-1
            model.layers[i].initialize(model.layers[i], mini_batch)
        end
    end

    function activate_ResNet(model::ResNet, data::Array{Float32})
        model.layers[1].output = data
        for i in 2:length(model.layers)-1
            model.layers[i].activate(model.layers[i], model.layers[i-1].output)
        end
    end

    function update_ResNet(model::ResNet, optimizer::String, α::Float64, parameters...)
        for i in length(model.layers)-1:-1:2
            model.layers[i].update(model.layers[i], optimizer, model.layers[i-1].output, model.layers[i+1].δ, α, parameters)
        end
    end
end