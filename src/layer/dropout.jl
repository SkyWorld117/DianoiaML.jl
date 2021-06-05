module DropoutM
    using LoopVectorization, VectorizedRNG

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

        function Dropout(;input_shape::Tuple, rate::Float64)
            new(save_Dropout, load_Dropout, activate_Dropout, init_Dropout, update_Dropout, input_shape, input_shape, rate)
        end
    end

    function init_Dropout(layer::Dropout, mini_batch::Int64)
        layer.table = zeros(Float32, layer.output_shape..., mini_batch)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end
    
    function activate_Dropout(layer::Dropout, input::Array{Float32})
        rand!(local_rng(), layer.table)
        @avxt for i in eachindex(input)
            layer.output[i] = ifelse(layer.table[i]<=layer.rate, 0.0f0, input[i])
        end
    end

    function update_Dropout(layer::Dropout, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=1)
        @avxt for i in eachindex(δₗ₊₁)
            layer.δ[i] = ifelse(layer.table[i]<=layer.rate, 0.0f0, δₗ₊₁[i])
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