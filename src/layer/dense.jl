module DenseM
    import ..DianoiaML: precision_type
    using ..DianoiaML: write_array, read_array
    using ..DianoiaML: checked_precision, accumulation_type
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer, load_layer!
    using ..GlorotUniform: GUMatrix
    using ..DianoiaML: LogitGradient, activation_gradient!, update_parameters!
    import ..DianoiaML: backward!, supports_backward, parameter_gradients
    using LinearAlgebra: mul!

    mutable struct Dense{T<:AbstractFloat, A<:AbstractFloat} <: AbstractLayer
        input_shape::Tuple{Vararg{Int}}
        output_shape::Tuple{Vararg{Int}}
        layer_size::Int
        activation_function::Module

        weights::Matrix{T}
        biases::Vector{T}

        Vdw::Matrix{A}
        Sdw::Matrix{A}
        Vdb::Vector{A}
        Sdb::Vector{A}
        weight_gradient::Matrix{A}
        bias_gradient::Vector{A}

        value::Matrix{T}
        output::Array{T}
        ∇biases::Matrix{T}
        δ::Array{T}

        function Dense(;input_shape::Tuple, layer_size::Int, activation_function::Module, randomization::Bool=true, precision::Type{T}=Float32) where {T<:AbstractFloat}
            checked_precision(T)
            A = accumulation_type(T)
            !isempty(input_shape) && all(>(0), input_shape) && layer_size > 0 ||
                throw(ArgumentError("Dense requires positive input and output sizes"))
            weight_shape = (layer_size, input_shape[1])
            weights = randomization ? GUMatrix(input_shape[1], layer_size, weight_shape, T) : zeros(T, weight_shape)
            biases = zeros(T, layer_size)

            output_shape = (layer_size, Base.tail(input_shape)...)

            Vdw = zeros(A, size(weights))
            Sdw = zeros(A, size(weights))
            Vdb = zeros(A, size(biases))
            Sdb = zeros(A, size(biases))
            new{T,A}(input_shape, output_shape, layer_size, activation_function,
                weights, biases, Vdw, Sdw, Vdb, Sdb,
                similar(weights, A), similar(biases, A))
        end
    end

    precision_type(::Dense{T}) where {T} = T

    function initialize!(layer::Dense{T}, mini_batch::Int) where {T}
        isdefined(layer, :δ) && size(layer.δ, ndims(layer.δ)) == mini_batch && return
        resize_buffers!(layer, (layer.input_shape..., mini_batch))
    end

    function resize_buffers!(layer::Dense{T}, input_size) where {T}
        output_size = (layer.layer_size, Base.tail(input_size)...)
        layer.value = zeros(T, layer.layer_size, prod(Base.tail(input_size)))
        layer.output = zeros(T, output_size)
        layer.∇biases = similar(layer.value)
        layer.δ = zeros(T, input_size)
    end

    function multiply!(dest, left, right, ::Type{A}, scale=one(A)) where {A}
        x = eltype(left) === A ? left : A.(left)
        y = eltype(right) === A ? right : A.(right)
        result = eltype(dest) === A ? dest : similar(dest, A)
        mul!(result, x, y, scale, zero(A))
        result === dest || copyto!(dest, result)
        return dest
    end

    function activate!(layer::Dense{T,A}, input::Array{T}) where {T,A}
        ndims(input) >= 2 && size(input, 1) == size(layer.weights, 2) ||
            throw(DimensionMismatch("Dense expects features on the first axis and examples on the last axis"))
        if !isdefined(layer, :δ) || size(layer.δ) != size(input)
            resize_buffers!(layer, size(input))
        end
        multiply!(layer.value, layer.weights, reshape(input, size(input, 1), :), A)
        layer.value .+= layer.biases
        layer.activation_function.func!(reshape(layer.output, size(layer.value)), layer.value)
        return layer.output
    end

    supports_backward(::Dense) = true
    parameter_gradients(layer::Dense) =
        (; weights=(layer.weights, layer.weight_gradient), biases=(layer.biases, layer.bias_gradient))

    function backward!(layer::Dense{T,A}, input::Array{T}, upstream; compute_parameters=true) where {T,A}
        delta = layer.∇biases
        data = upstream isa LogitGradient ? upstream.data : upstream
        size(data) == size(layer.output) || throw(DimensionMismatch("Dense cotangent shape differs from its output"))
        matrix_delta = reshape(data, size(delta))
        activation_gradient!(delta, layer.activation_function, layer.value,
                             upstream isa LogitGradient ? LogitGradient(matrix_delta) : matrix_delta)
        multiply!(reshape(layer.δ, size(input, 1), :), transpose(layer.weights), delta, A)
        compute_parameters || return layer.δ

        batch = size(input, ndims(input))
        input_matrix = reshape(input, size(input, 1), :)
        multiply!(layer.weight_gradient, delta, transpose(input_matrix), A, inv(A(batch)))
        fill!(layer.bias_gradient, zero(A))
        for b in axes(delta, 2)
            @inbounds @simd for i in axes(delta, 1)
                layer.bias_gradient[i] += A(delta[i, b])
            end
        end
        layer.bias_gradient ./= A(batch)
        return layer.δ
    end

    function update!(layer::Dense{T}, optimizer::String, input::Array{T}, delta,
                     α::Real, parameters::Tuple, direction::Int=1) where {T}
        backward!(layer, input, delta; compute_parameters=direction != 0)
        direction == 0 && return
        rate = accumulation_type(T)(α * direction)
        update_parameters!(optimizer, layer.weights, layer.weight_gradient, layer.Vdw, layer.Sdw, rate, parameters)
        update_parameters!(optimizer, layer.biases, layer.bias_gradient, layer.Vdb, layer.Sdb, rate, parameters)
    end

    function save_layer(layer::Dense, file, id::String)
        write(file, id, "Dense")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"layer_size", layer.layer_size)
        write(file, id*"activation_function", layer.activation_function.get_name())

        write_array(file, id*"weights", layer.weights)
        write_array(file, id*"biases", layer.biases)
    end

    function load_layer!(layer::Dense, file, id::String)
        layer.weights = read_array(file, id*"weights")
        layer.biases = read_array(file, id*"biases")
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        layer_size = read(file, id*"layer_size")
        return (input_shape=input_shape, layer_size=layer_size)
    end
end
