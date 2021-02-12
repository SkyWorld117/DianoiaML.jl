module dense
    include("../tools/glorotuniform.jl")
    using .GlorotUniform

    mutable struct Dense
        save_layer::Any
        load_layer::Any
        activator::Any
        get_PU::Any
        update_weights::Bool
        update_biases::Bool

        input_size::Int64
        layer_size::Int64
        activation_function::Module

        weights::Array{Float32}
        biases::Array{Float32}
        weights_prop::Array{Int8}
        biases_prop::Array{Int8}

        Vdw::Array{Float32}
        Sdw::Array{Float32}
        Vdb::Array{Float32}
        Sdb::Array{Float32}

        value::Array{Float32}
        output::Array{Float32}
        propagation_units::Array{Float32}

        function Dense(;input_size::Int64, layer_size::Int64, activation_function::Module, randomization::Bool=true, reload::Bool=false)
            if reload
                return new(save_Dense, load_Dense, activate_Dense, PU_Dense, true, true)
            end

            if randomization
                weights = Array{Float32}(GUMatrix(input_size, layer_size, (layer_size, input_size)))
                biases = Array{Float32}(GUMatrix(input_size, layer_size, layer_size))
            else
                weights = zeros(Float32, (layer_size, input_size))
                biases = zeros(Float32, layer_size)
            end
            weights_prop = ones(Int8, size(weights))
            biases_prop = ones(Int8, size(biases))

            Vdw = zeros(Float32, size(weights))
            Sdw = zeros(Float32, size(weights))
            Vdb = zeros(Float32, size(biases))
            Sdb = zeros(Float32, size(biases))
            new(save_Dense, load_Dense, activate_Dense, PU_Dense, true, true, input_size, layer_size, activation_function, weights, biases, weights_prop, biases_prop, Vdw, Sdw, Vdb, Sdb)
        end
    end

    function activate_Dense(layer::Dense, input::Array{Float32})
        layer.value = layer.weights*input .+ layer.biases
        layer.output = layer.activation_function.func(layer.value)
    end

    function PU_Dense(layer::Dense, ∇biases::Array{Float32})
        layer.propagation_units = transpose(layer.weights)*∇biases
    end

    function save_Dense(layer::Dense, file::Any, id::Int64)
        write(file, string(id), "Dense")
        write(file, string(id)*"weights", layer.weights)
        write(file, string(id)*"biases", layer.biases)
        write(file, string(id)*"activation_function", layer.activation_function.get_name())
    end

    function load_Dense(layer::Dense, file::Any, id::Int64)
        layer.weights = read(file, string(id)*"weights")
        layer.biases = read(file, string(id)*"biases")
        layer.input_size = size(layer.weights, 2)
        layer.layer_size = size(layer.weights, 1)
        layer.weights_prop = ones(Int8, size(layer.weights))
        layer.biases_prop = ones(Int8, size(layer.biases))
        layer.Vdw = zeros(Float32, size(layer.weights))
        layer.Sdw = zeros(Float32, size(layer.weights))
        layer.Vdb = zeros(Float32, size(layer.biases))
        layer.Sdb = zeros(Float32, size(layer.biases))
    end
end
