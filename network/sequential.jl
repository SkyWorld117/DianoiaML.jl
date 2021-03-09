module sequential
    using .Threads

    mutable struct Sequential
        layers::Array{Any}
        add_layer::Any
        activate::Any
        initialize::Any
        update::Any
        num_layer::Int64

        loss::Float64

        default_input_size::Int64
        default_input_filter::Int64
        default_input2D_size::Tuple
        function Sequential()
            new(Any[Hidden_Input_Layer(Float32[])], Sequential_add_layer, activate_Sequential, init_Sequential, update_Sequential, 0, 0.0, 0, 1, ())
        end
    end

    mutable struct Hidden_Input_Layer
        output::Array{Float32}
    end
    mutable struct Hidden_Output_Layer
        propagation_units::Array{Float32}
    end

    function Sequential_add_layer(model::Sequential, layer::Any; args...)
        try
            push!(model.layers, layer(;input_size=model.default_input_size, input_filter=model.default_input_filter, input2D_size=model.default_input2D_size, args...))
        catch e
            push!(model.layers, layer(;input_size=model.default_input_size, args...))
        end
        model.num_layer += 1
        model.default_input_size = model.layers[end].layer_size
        try
            kwargs = Dict(args)
            model.default_input_filter = kwargs[:filter]
            input2D_size = model.layers[end].input2D_size
            #padding = kwargs[:padding]
            kernel_size = kwargs[:kernel_size]
            model.default_input2D_size = (input2D_size[1]-kernel_size[1]+1, input2D_size[2]-kernel_size[2]+1)
        catch e
        end
    end

    function init_Sequential(model::Sequential, mini_batch::Int64)
        if length(model.layers)==model.num_layer+1
            push!(model.layers, Hidden_Output_Layer(Float32[]))
        end
        @threads for i in 2:length(model.layers)-1
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
