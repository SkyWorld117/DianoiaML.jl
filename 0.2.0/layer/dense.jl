module dense
    mutable struct Dense
        activator::Any

        input_size::Int64
        layer_size::Int64
        activation_function::Module

        weights::Array{Float32}
        biases::Array{Float32}

        value::Array{Float32}
        output::Array{Float32}
        propagation_units::Array{Float32}

        function Dense(;input_size::Int64, layer_size::Int64, randomization::Bool, activation_function::Module, reload::Bool=false)
            if reload
                return new(activate_Dense)
            end

            if randomization
                weights = 0.1*rand(Float32, (layer_size, input_size))
                biases = 0.1*rand(Float32, layer_size)
            else
                weights = zeros(Float32, (layer_size, input_size))
                biases = zeros(Float32, layer_size)
            end
            new(activate_Dense, input_size, layer_size, activation_function, weights, biases)
        end
    end

    function activate_Dense(layer::Dense, input::Array{Float32})
        layer.value = layer.weights*input .+ layer.biases
        layer.output = layer.activation_function.func(layer.value)
        layer.propagation_units = zeros((layer.layer_size, layer.input_size, size(input, 2)))
    end
end
