module DenseM
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer, load_layer!
    using ..GlorotUniform: GUMatrix
    using ..DianoiaML: LogitGradient, activation_gradient!, update_parameters!
    import ..DianoiaML: backward!, supports_backward, parameter_gradients
    using LinearAlgebra: mul!

    mutable struct Dense <: AbstractLayer
        input_shape::Tuple{Vararg{Int}}
        output_shape::Tuple{Vararg{Int}}
        layer_size::Int
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
        output::Array{Float32}
        ∇biases::Matrix{Float32}
        δ::Array{Float32}

        function Dense(;input_shape::Tuple, layer_size::Int, activation_function::Module, randomization::Bool=true)
            !isempty(input_shape) && all(>(0), input_shape) && layer_size > 0 ||
                throw(ArgumentError("Dense requires positive input and output sizes"))
            weight_shape = (layer_size, input_shape[1])
            weights = randomization ? GUMatrix(input_shape[1], layer_size, weight_shape) : zeros(Float32, weight_shape)
            biases = zeros(Float32, layer_size)

            output_shape = (layer_size, Base.tail(input_shape)...)

            Vdw = zeros(Float32, size(weights))
            Sdw = zeros(Float32, size(weights))
            Vdb = zeros(Float32, size(biases))
            Sdb = zeros(Float32, size(biases))
            new(input_shape, output_shape, layer_size, activation_function,
                weights, biases, Vdw, Sdw, Vdb, Sdb,
                similar(weights), similar(biases))
        end
    end

    function initialize!(layer::Dense, mini_batch::Int)
        isdefined(layer, :δ) && size(layer.δ, ndims(layer.δ)) == mini_batch && return
        resize_buffers!(layer, (layer.input_shape..., mini_batch))
    end

    function resize_buffers!(layer::Dense, input_size)
        output_size = (layer.layer_size, Base.tail(input_size)...)
        layer.value = zeros(Float32, layer.layer_size, prod(Base.tail(input_size)))
        layer.output = zeros(Float32, output_size)
        layer.∇biases = similar(layer.value)
        layer.δ = zeros(Float32, input_size)
    end

    function activate!(layer::Dense, input::Array{Float32})
        ndims(input) >= 2 && size(input, 1) == size(layer.weights, 2) ||
            throw(DimensionMismatch("Dense expects features on the first axis and examples on the last axis"))
        if !isdefined(layer, :δ) || size(layer.δ) != size(input)
            resize_buffers!(layer, size(input))
        end
        mul!(layer.value, layer.weights, reshape(input, size(input, 1), :))
        layer.value .+= layer.biases
        layer.activation_function.func!(reshape(layer.output, size(layer.value)), layer.value)
        return layer.output
    end

    supports_backward(::Dense) = true
    parameter_gradients(layer::Dense) =
        (; weights=(layer.weights, layer.weight_gradient), biases=(layer.biases, layer.bias_gradient))

    function backward!(layer::Dense, input::Array{Float32}, upstream; compute_parameters=true)
        delta = layer.∇biases
        data = upstream isa LogitGradient ? upstream.data : upstream
        size(data) == size(layer.output) || throw(DimensionMismatch("Dense cotangent shape differs from its output"))
        matrix_delta = reshape(data, size(delta))
        activation_gradient!(delta, layer.activation_function, layer.value,
                             upstream isa LogitGradient ? LogitGradient(matrix_delta) : matrix_delta)
        mul!(reshape(layer.δ, size(input, 1), :), transpose(layer.weights), delta)
        compute_parameters || return layer.δ

        batch = size(input, ndims(input))
        input_matrix = reshape(input, size(input, 1), :)
        mul!(layer.weight_gradient, delta, transpose(input_matrix), inv(Float32(batch)), 0f0)
        fill!(layer.bias_gradient, 0f0)
        for b in axes(delta, 2)
            @inbounds @simd for i in axes(delta, 1)
                layer.bias_gradient[i] += delta[i, b]
            end
        end
        layer.bias_gradient ./= Float32(batch)
        return layer.δ
    end

    function update!(layer::Dense, optimizer::String, input::Array{Float32}, delta,
                     α::Float64, parameters::Tuple, direction::Int=1)
        backward!(layer, input, delta; compute_parameters=direction != 0)
        direction == 0 && return
        rate = Float32(α * direction)
        update_parameters!(optimizer, layer.weights, layer.weight_gradient, layer.Vdw, layer.Sdw, rate, parameters)
        update_parameters!(optimizer, layer.biases, layer.bias_gradient, layer.Vdb, layer.Sdb, rate, parameters)
    end

    function save_layer(layer::Dense, file, id::String)
        write(file, id, "Dense")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"layer_size", layer.layer_size)
        write(file, id*"activation_function", layer.activation_function.get_name())

        write(file, id*"weights", layer.weights)
        write(file, id*"biases", layer.biases)
    end

    function load_layer!(layer::Dense, file, id::String)
        layer.weights = read(file, id*"weights")
        layer.biases = read(file, id*"biases")
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        layer_size = read(file, id*"layer_size")
        return (input_shape=input_shape, layer_size=layer_size)
    end
end
