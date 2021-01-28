module sequential
    mutable struct Sequential
        layers::Array{Any}
        add_layer::Any
        activator::Any
        num_layer::Int64

        function Sequential()
            new(Any[Hidden_Input_Layer(Float32[])], Sequential_add_layer, activate_Sequential, 0)
        end
    end

    mutable struct Hidden_Input_Layer
        output::Array{Float32}
    end
    mutable struct Hidden_Output_Layer
        propagation_units::Array{Float32}
    end

    function Sequential_add_layer(model::Sequential, layer::Any)
        push!(model.layers, layer)
        model.num_layer += 1
    end

    function activate_Sequential(model::Sequential, data::Array{Float32})
        model.layers[1].output = data
        if length(model.layers)==model.num_layer+1
            push!(model.layers, Hidden_Output_Layer(Float32[]))
        end
        for i in 2:length(model.layers)-1
            model.layers[i].activator(model.layers[i], model.layers[i-1].output)
        end
        return model.layers[end-1].output
    end
end
