module DropoutM
    import ..DianoiaML: precision_type
    using ..DianoiaML: checked_precision, accumulation_type, random_precision, layer_input,
                       read_number, write_number
    import ..DianoiaML: backward!, supports_backward
    using Random: rand!, default_rng
    using ..DianoiaML: TrainingMode
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer, activate_layer

    mutable struct Dropout{T<:AbstractFloat,N,R<:AbstractFloat,A<:AbstractFloat} <: AbstractLayer
        input_shape::Tuple
        output_shape::Tuple

        rate::R
        table::Array{A,N}

        output::Array{T,N}
        δ::Array{T,N}
        training::Bool

        function Dropout(;input_shape::Tuple, rate::Real, precision::Type{T}=Float32) where {T<:AbstractFloat}
            checked_precision(T)
            A = accumulation_type(T)
            R = promote_type(typeof(float(rate)), A)
            threshold = convert(R, rate)
            0 <= threshold < 1 || throw(ArgumentError("dropout rate must be in [0, 1)"))
            new{T,length(input_shape) + 1,R,random_precision(A)}(input_shape, input_shape, threshold)
        end
    end

    precision_type(::Dropout{T}) where {T} = T

    function initialize!(layer::Dropout{T,N,R,A}, mini_batch::Int) where {T,N,R,A}
        isdefined(layer, :table) && size(layer.table, ndims(layer.table)) == mini_batch && return
        layer.table = zeros(A, layer.output_shape..., mini_batch)
        layer.output = zeros(T, layer.output_shape..., mini_batch)
        layer.δ = zeros(T, layer.input_shape..., mini_batch)
        layer.training = false
    end

    activate_layer(layer::Dropout, input, training) = activate!(layer, layer_input(layer, input); training)

    activate_layer(layer::Dropout, input, mode::TrainingMode) =
        activate!(layer, layer_input(layer, input); training=true, rng=mode.rng)

    function activate!(layer::Dropout{T,N,R,A}, input::Array{T};
                       training::Bool=true, rng=default_rng()) where {T,N,R,A}
        ndims(input) == length(layer.input_shape) + 1 ||
            throw(DimensionMismatch("unexpected dropout input rank"))
        if !isdefined(layer, :output) || size(input) != size(layer.output)
            layer.table = similar(input, A)
            layer.output = similar(input)
            layer.δ = similar(input)
        end
        layer.training = training
        if training
            rand!(rng, layer.table)
            apply_mask!(layer.output, layer.table, input, layer.rate)
        else
            copyto!(layer.output, input)
        end
    end

    supports_backward(::Dropout) = true

    function apply_mask!(dest, table, data, rate)
        scale = accumulation_type(eltype(dest))(inv(1 - rate))
        dest .= ifelse.(table .< eltype(table)(rate), zero(eltype(dest)), data .* scale)
    end

    function backward!(layer::Dropout, input, δₗ₊₁)
        if layer.training
            apply_mask!(layer.δ, layer.table, δₗ₊₁, layer.rate)
        else
            copyto!(layer.δ, δₗ₊₁)
        end
        return layer.δ
    end

    function update!(layer::Dropout{T}, optimizer::String, input, delta, α::Real, parameters::Tuple, direction::Int=1) where {T}
        backward!(layer, input, delta)
    end

    function save_layer(layer::Dropout, file, id::String)
        write(file, id, "Dropout")
        write(file, id*"input_shape", collect(layer.input_shape))
        write_number(file, id*"rate", layer.rate)
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        rate = read_number(file, id*"rate")
        return (input_shape=input_shape, rate=rate)
    end
end
