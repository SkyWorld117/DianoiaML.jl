module DropoutM
    using Random: rand!
    import ..DianoiaML: activate_layer

    mutable struct Dropout
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple

        rate::Float64
        table::Array{Float32}

        output::Array{Float32}
        δ::Array{Float32}
        training::Bool

        function Dropout(;input_shape::Tuple, rate::Float64)
            0 <= rate < 1 || throw(ArgumentError("dropout rate must be in [0, 1)"))
            new(save_Dropout, load_Dropout, activate_Dropout, init_Dropout, update_Dropout, input_shape, input_shape, rate)
        end
    end

    function init_Dropout(layer::Dropout, mini_batch::Int64)
        isdefined(layer, :table) && size(layer.table, ndims(layer.table)) == mini_batch && return
        layer.table = zeros(Float32, layer.output_shape..., mini_batch)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
        layer.training = false
    end

    activate_layer(layer::Dropout, input, training) = activate_Dropout(layer, input; training)

    function activate_Dropout(layer::Dropout, input::Array{Float32}; training::Bool=true)
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

    function update_Dropout(layer::Dropout, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int=1)
        if layer.training
            scale = Float32(inv(1 - layer.rate))
            layer.δ .= ifelse.(layer.table .< Float32(layer.rate), 0f0, δₗ₊₁ .* scale)
        else
            copyto!(layer.δ, δₗ₊₁)
        end
    end

    function save_Dropout(layer::Dropout, file::Any, id::String)
        write(file, id, "Dropout")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"rate", layer.rate)
    end

    function load_Dropout(layer::Dropout, file::Any, id::String)
    end

    function get_args(file::Any, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        rate = read(file, id*"rate")
        return (input_shape=input_shape, rate=rate)
    end
end
