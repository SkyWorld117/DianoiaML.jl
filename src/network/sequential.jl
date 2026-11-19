module sequential
    using ..DianoiaML: activate_layer
    import ..DianoiaML

    mutable struct Sequential
        layers::Array{Any}
        add_layer::Any
        activate::Any
        initialize::Any
        update::Any
        num_layer::Int64

        loss::Float64

        default_input_shape::Tuple

        function Sequential()
            new(Any[Hidden_Input_Layer(Float32[])], Sequential_add_layer, activate_Sequential, init_Sequential, update_Sequential, 0, 0.0, ())
        end
    end

    mutable struct Hidden_Input_Layer
        output::Array{Float32}
    end
    mutable struct Hidden_Output_Layer
        δ::Array{Float32}
    end

    function Sequential_add_layer(model::Sequential, layer::Any; args...)
        next = layer(; input_shape=model.default_input_shape, args...)
        insert!(model.layers, model.num_layer + 2, next)
        model.num_layer += 1
        model.default_input_shape = next.output_shape
    end

    function init_Sequential(model::Sequential, mini_batch::Int)
        model.num_layer > 0 || throw(ArgumentError("Sequential requires at least one layer"))
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

    function activate_Sequential(model::Sequential, data::Array{Float32}; training::Bool=false)
        model.layers[1].output = data
        for i in 2:model.num_layer+1
            layer = model.layers[i]
            activate_layer(layer, model.layers[i-1].output, training)
        end
        return model.layers[model.num_layer+1].output
    end

    function update_Sequential(model::Sequential, current_input_data::Array{Float32}, current_output_data::Array{Float32}, loss_function::Any, monitor::Any, optimizer::String, α::Float64, parameters...)
        model.activate(model, current_input_data; training=true)
        delta = DianoiaML.output_gradient!(model.layers[end].δ, model.layers[end-1], current_output_data, loss_function)
        for i in length(model.layers)-1:-1:2
            layer = model.layers[i]
            layer.update(layer, optimizer, model.layers[i-1].output, delta, α, parameters)
            delta = layer.δ
        end
        model.loss += monitor.func(model.layers[end-1].output, current_output_data)
    end
end
