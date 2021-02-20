module sequential
    def = 0

    mutable struct Sequential
        layers::Array{Any}
        add_layer::Any
        activate::Any
        initialize::Any
        update::Any
        num_layer::Int64

        loss::Float64

        function Sequential()
            new(Any[Hidden_Input_Layer(Float32[])], Sequential_add_layer, activate_Sequential, init_Sequential, update_Sequential, 0, 0.0)
        end
    end

    mutable struct Hidden_Input_Layer
        output::Array{Float32}
    end
    mutable struct Hidden_Output_Layer
        propagation_units::Array{Float32}
    end

    function Sequential_add_layer(model::Sequential, layer::Any)
        global def
        push!(model.layers, layer)
        model.num_layer += 1
        def = layer.layer_size
    end

    function init_Sequential(model::Sequential, mini_batch::Int64)
        if length(model.layers)==model.num_layer+1
            push!(model.layers, Hidden_Output_Layer(Float32[]))
        end
        Threads.@threads for i in 2:length(model.layers)-1
            model.layers[i].initialize(model.layers[i], mini_batch)
        end
    end

    function activate_Sequential(model::Sequential, data::Array{Float32})
        model.layers[1].output = data
        for i in 2:length(model.layers)-1
            model.layers[i].activate(model.layers[i], model.layers[i-1].output)
        end
        return model.layers[end-1].output
    end

    function update_Sequential(model::Sequential, current_input_data::Array{Float32}, current_output_data::Array{Float32}, loss_function::Any, monitor::Any, optimizer::String, α::Float64, parameters...)
        model.activate(model, current_input_data)
        model.layers[end].propagation_units = loss_function.prop(model.layers[end-1].output, current_output_data)
        for i in length(model.layers)-1:-1:2
            model.layers[i].update(model.layers[i], optimizer, model.layers[i-1].output, model.layers[i+1].propagation_units, α, parameters)
        end
        model.loss += monitor.func(model.layers[end-1].output, current_output_data)
    end
end
