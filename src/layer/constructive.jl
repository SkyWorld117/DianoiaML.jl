module ConstructiveM
    using LoopVectorization

    mutable struct Constructive
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple

        output::Array{Float32}
        δ::Array{Float32}

        function Constructive(;input_shape::Tuple, shape::Tuple)
            new(save_Constructive, load_Constructive, activate_Constructive, init_Constructive, update_Constructive, input_shape, shape)
        end
    end

    function init_Constructive(layer::Constructive, mini_batch::Int64)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate_Constructive(layer::Constructive, input::Array{Float32})
        @avxt for i in eachindex(layer.output)
            layer.output[i] = input[i]
        end
    end

    function update_Constructive(layer::Constructive, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=1)
        @avxt for i in eachindex(δₗ₊₁)
            layer.δ[i] = δₗ₊₁[i]
        end
    end

    function save_Constructive(layer::Constructive, file::Any, id::Int64)
        write(file, string(id), "Constructive")
        write(file, string(id)*"input_shape", collect(layer.input_shape))
        write(file, string(id)*"shape", collect(layer.output_shape))
    end

    function load_Constructive(layer::Constructive, file::Any, id::Int64)
    end

    function get_args(file::Any, id::Int64)
        input_shape = tuple(read(file, string(id)*"input_shape")...)
        shape = tuple(read(file, string(id)*"shape")...)
        return (input_shape=input_shape, shape=shape)
    end
end