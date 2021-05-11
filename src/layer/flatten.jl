module FlattenM
    using LoopVectorization

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

        function Flatten(;input_shape)
            s = 1
            for i in input_shape
                s *= i
            end
            output_shape = (s,)

            new(save_Flatten, load_Flatten, activate_Flatten, init_Flatten, update_Flatten, input_shape, output_shape)
        end
    end

    function init_Flatten(layer::Flatten, mini_batch)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate_Flatten(layer::Flatten, input)
        @avxt for i in eachindex(input)
            layer.output[i] = input[i]
        end
    end

    function update_Flatten(layer::Flatten, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=1)
        @avxt for i in eachindex(δₗ₊₁)
            layer.δ[i] = δₗ₊₁[i]
        end
    end

    function save_Flatten(layer::Flatten, file::Any, id::Int64)
        write(file, string(id), "Flatten")
        write(file, string(id)*"input_shape", collect(layer.input_shape))
    end

    function load_Flatten(layer::Flatten, file::Any, id::Int64)
    end

    function get_args(file::Any, id::Int64)
        input_shape = tuple(read(file, string(id)*"input_shape")...)
        return (input_shape=input_shape,)
    end
end
