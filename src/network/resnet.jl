module resnet
    using ..DianoiaML: activate_layer

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
        next = layer(; input_shape=model.default_input_shape, args...)
        insert!(model.layers, model.num_layer + 2, next)
        model.num_layer += 1
        model.default_input_shape = next.output_shape
    end

    function init_ResNet(model::ResNet, mini_batch::Int)
        model.num_layer > 0 || throw(ArgumentError("ResNet requires at least one layer"))
        mini_batch > 0 || throw(ArgumentError("batch size must be positive"))
        output_index = model.num_layer + 2
        output_shape = (model.default_input_shape..., mini_batch)
        if length(model.layers) < output_index
            push!(model.layers, Hidden_Output_Layer(zeros(Float32, output_shape)))
        elseif size(model.layers[end].δ) != output_shape
            model.layers[end].δ = zeros(Float32, output_shape)
        end
        for i in 2:output_index-1
            model.layers[i].initialize(model.layers[i], mini_batch)
        end
    end

    function activate_ResNet(model::ResNet, data::Array{Float32}; training::Bool=false)
        model.layers[1].output = data
        for i in 2:model.num_layer+1
            layer = model.layers[i]
            activate_layer(layer, model.layers[i-1].output, training)
        end
        return model.layers[model.num_layer+1].output
    end

    function update_ResNet(model::ResNet, optimizer::String, α::Float64, parameters...; direction::Int=1)
        for i in length(model.layers)-1:-1:2
            layer = model.layers[i]
            input = model.layers[i-1].output
            delta = model.layers[i+1].δ
            if direction == 1
                layer.update(layer, optimizer, input, delta, α, parameters)
            else
                layer.update(layer, optimizer, input, delta, α, parameters, direction)
            end
        end
    end
end
