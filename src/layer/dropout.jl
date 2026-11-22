module DropoutM
    using Random: rand!
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

    function activate!(layer::Dropout, input::Array{Float32}; training::Bool=true)
        size(input) == size(layer.output) || throw(DimensionMismatch("unexpected dropout input size"))
        layer.training = training
        if training
            rand!(layer.table)
            scale = Float32(inv(1 - layer.rate))
            layer.output .= ifelse.(layer.table .< Float32(layer.rate), 0f0, input .* scale)
        else
            copyto!(layer.output, input)
        end
    end

    function update!(layer::Dropout, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int=1)
        if layer.training
            scale = Float32(inv(1 - layer.rate))
            layer.δ .= ifelse.(layer.table .< Float32(layer.rate), 0f0, δₗ₊₁ .* scale)
        else
            copyto!(layer.δ, δₗ₊₁)
        end
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
