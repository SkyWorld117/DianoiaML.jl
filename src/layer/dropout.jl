module DropoutM
    import ..DianoiaML: backward!, supports_backward
    using Random: rand!, default_rng
    using ..DianoiaML: TrainingMode
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer, activate_layer

    mutable struct Dropout{N} <: AbstractLayer
        input_shape::Tuple
        output_shape::Tuple

        rate::Float64
        table::Array{Float32,N}

        output::Array{Float32,N}
        δ::Array{Float32,N}
        training::Bool

        function Dropout(;input_shape::Tuple, rate::Float64)
            0 <= rate < 1 || throw(ArgumentError("dropout rate must be in [0, 1)"))
            new{length(input_shape) + 1}(input_shape, input_shape, rate)
        end
    end

    function initialize!(layer::Dropout, mini_batch::Int)
        isdefined(layer, :table) && size(layer.table, ndims(layer.table)) == mini_batch && return
        layer.table = zeros(Float32, layer.output_shape..., mini_batch)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
        layer.training = false
    end

    activate_layer(layer::Dropout, input, training) = activate!(layer, input; training)

    activate_layer(layer::Dropout, input, mode::TrainingMode) =
        activate!(layer, input; training=true, rng=mode.rng)

    function activate!(layer::Dropout, input::Array{Float32}; training::Bool=true, rng=default_rng())
        ndims(input) == length(layer.input_shape) + 1 ||
            throw(DimensionMismatch("unexpected dropout input rank"))
        if !isdefined(layer, :output) || size(input) != size(layer.output)
            layer.table = similar(input)
            layer.output = similar(input)
            layer.δ = similar(input)
        end
        layer.training = training
        if training
            rand!(rng, layer.table)
            scale = Float32(inv(1 - layer.rate))
            layer.output .= ifelse.(layer.table .< Float32(layer.rate), 0f0, input .* scale)
        else
            copyto!(layer.output, input)
        end
    end

    supports_backward(::Dropout) = true

    function backward!(layer::Dropout, input, δₗ₊₁)
        if layer.training
            scale = Float32(inv(1 - layer.rate))
            layer.δ .= ifelse.(layer.table .< Float32(layer.rate), 0f0, δₗ₊₁ .* scale)
        else
            copyto!(layer.δ, δₗ₊₁)
        end
        return layer.δ
    end

    function update!(layer::Dropout, optimizer::String, input, delta, α::Float64, parameters::Tuple, direction::Int=1)
        backward!(layer, input, delta)
    end

    function save_layer(layer::Dropout, file, id::String)
        write(file, id, "Dropout")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"rate", layer.rate)
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        rate = read(file, id*"rate")
        return (input_shape=input_shape, rate=rate)
    end
end
