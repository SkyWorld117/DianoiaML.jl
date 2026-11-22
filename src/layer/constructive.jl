module ConstructiveM
    import ..DianoiaML: precision_type
    using ..DianoiaML: checked_precision
    import ..DianoiaML: backward!, supports_backward
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer
    mutable struct Constructive{T<:AbstractFloat, N,M} <: AbstractLayer
        input_shape::Tuple
        output_shape::Tuple

        output::Array{T,M}
        δ::Array{T,N}

        function Constructive(;input_shape::Tuple, shape::Tuple, precision::Type{T}=Float32) where {T<:AbstractFloat}
            checked_precision(T)
            prod(input_shape) == prod(shape) || throw(DimensionMismatch("reshape changes the feature count"))
            new{T,length(input_shape) + 1, length(shape) + 1}(input_shape, shape)
        end
    end

    precision_type(::Constructive{T}) where {T} = T

    function initialize!(layer::Constructive{T}, mini_batch::Int) where {T}
        isdefined(layer, :output) && size(layer.output, ndims(layer.output)) == mini_batch && return
        layer.output = zeros(T, layer.output_shape..., mini_batch)
        layer.δ = zeros(T, layer.input_shape..., mini_batch)
    end

    function activate!(layer::Constructive{T}, input::Array{T}) where {T}
        length(layer.output) == length(input) || throw(DimensionMismatch("unexpected reshape input size"))
        copyto!(layer.output, input)
    end

    supports_backward(::Constructive) = true

    function backward!(layer::Constructive{T}, input, δₗ₊₁) where {T}
        length(layer.δ) == length(δₗ₊₁) || throw(DimensionMismatch("unexpected reshape gradient size"))
        copyto!(layer.δ, δₗ₊₁)
        return layer.δ
    end

    function update!(layer::Constructive{T}, optimizer::String, input, delta, α::Real, parameters::Tuple, direction::Int=1) where {T}
        backward!(layer, input, delta)
    end

    function save_layer(layer::Constructive, file, id::String)
        write(file, id, "Constructive")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"shape", collect(layer.output_shape))
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        shape = tuple(read(file, id*"shape")...)
        return (input_shape=input_shape, shape=shape)
    end
end
