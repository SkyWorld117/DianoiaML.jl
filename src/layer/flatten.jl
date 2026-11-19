module FlattenM
    mutable struct Flatten
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple

        output::Array{Float32}
        δ::Array{Float32}

        function Flatten(;input_shape::Tuple)
            output_shape = (prod(input_shape),)

            new(save_Flatten, load_Flatten, activate_Flatten, init_Flatten, update_Flatten, input_shape, output_shape)
        end
    end

    function init_Flatten(layer::Flatten, mini_batch::Int64)
        isdefined(layer, :output) && size(layer.output, ndims(layer.output)) == mini_batch && return
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate_Flatten(layer::Flatten, input::Array{Float32})
        length(layer.output) == length(input) || throw(DimensionMismatch("unexpected reshape input size"))
        copyto!(layer.output, input)
    end

    function update_Flatten(layer::Flatten, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int=1)
        length(layer.δ) == length(δₗ₊₁) || throw(DimensionMismatch("unexpected reshape gradient size"))
        copyto!(layer.δ, δₗ₊₁)
    end

    function save_Flatten(layer::Flatten, file::Any, id::String)
        write(file, id, "Flatten")
        write(file, id*"input_shape", collect(layer.input_shape))
    end

    function load_Flatten(layer::Flatten, file::Any, id::String)
    end

    function get_args(file::Any, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        return (input_shape=input_shape,)
    end
end
