module DenseM
    using ..GlorotUniform: GUMatrix
    using ..DianoiaML: activation_gradient!, update_parameters!
    using LinearAlgebra: mul!

    mutable struct Dense
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple
        layer_size::Int64
        activation_function::Module

        weights::Matrix{Float32}
        biases::Vector{Float32}

        Vdw::Matrix{Float32}
        Sdw::Matrix{Float32}
        Vdb::Vector{Float32}
        Sdb::Vector{Float32}
        weight_gradient::Matrix{Float32}
        bias_gradient::Vector{Float32}

        value::Matrix{Float32}
        output::Matrix{Float32}
        ∇biases::Matrix{Float32}
        δ::Matrix{Float32}

        function Dense(;input_shape::Tuple, layer_size::Int64, activation_function::Module, randomization::Bool=true)
            length(input_shape) == 1 && input_shape[1] > 0 && layer_size > 0 ||
                throw(ArgumentError("Dense requires positive one-dimensional input and output sizes"))
            weight_shape = (layer_size, input_shape[1])
            weights = randomization ? GUMatrix(input_shape[1], layer_size, weight_shape) : zeros(Float32, weight_shape)
            biases = zeros(Float32, layer_size)

            output_shape = (layer_size,)

            Vdw = zeros(Float32, size(weights))
            Sdw = zeros(Float32, size(weights))
            Vdb = zeros(Float32, size(biases))
            Sdb = zeros(Float32, size(biases))
            new(save_Dense, load_Dense, activate_Dense, init_Dense, update_Dense, input_shape,
                output_shape, layer_size, activation_function, weights, biases, Vdw, Sdw, Vdb, Sdb,
                similar(weights), similar(biases))
        end
    end

    function init_Dense(layer::Dense, mini_batch::Int)
        isdefined(layer, :value) && size(layer.value, 2) == mini_batch && return
        layer.value = zeros(Float32, layer.layer_size, mini_batch)
        layer.output = similar(layer.value)
        layer.∇biases = similar(layer.value)
        layer.δ = zeros(Float32, layer.input_shape[1], mini_batch)
    end

    function activate_Dense(layer::Dense, input::Matrix{Float32})
        mul!(layer.value, layer.weights, input)
        layer.value .+= layer.biases
        layer.activation_function.func!(layer.output, layer.value)
    end

    function update_Dense(layer::Dense, optimizer::String, input::Matrix{Float32}, δₗ₊₁, α::Float64, parameters::Tuple, direction::Int=1)
        delta = layer.∇biases
        activation_gradient!(delta, layer.activation_function, layer.value, δₗ₊₁)
        mul!(layer.δ, transpose(layer.weights), delta)
        direction == 0 && return

        rate = Float32(α * direction)
        batch = size(input, 2)
        mul!(layer.weight_gradient, delta, transpose(input), inv(Float32(batch)), 0f0)
        fill!(layer.bias_gradient, 0f0)
        for b in axes(delta, 2)
            @inbounds @simd for i in axes(delta, 1)
                layer.bias_gradient[i] += delta[i, b]
            end
        end
        layer.bias_gradient ./= Float32(batch)
        update_parameters!(optimizer, layer.weights, layer.weight_gradient, layer.Vdw, layer.Sdw, rate, parameters)
        update_parameters!(optimizer, layer.biases, layer.bias_gradient, layer.Vdb, layer.Sdb, rate, parameters)
    end

    function save_Dense(layer::Dense, file::Any, id::String)
        write(file, id, "Dense")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"layer_size", layer.layer_size)
        write(file, id*"activation_function", layer.activation_function.get_name())

        write(file, id*"weights", layer.weights)
        write(file, id*"biases", layer.biases)
    end

    function load_Dense(layer::Dense, file::Any, id::String)
        layer.weights = read(file, id*"weights")
        layer.biases = read(file, id*"biases")
    end

    function get_args(file::Any, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        layer_size = read(file, id*"layer_size")
        return (input_shape=input_shape, layer_size=layer_size)
    end
end
