module FlattenM
    import ..DianoiaML: backward!, supports_backward
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer
    mutable struct Flatten{N} <: AbstractLayer
        input_shape::Tuple
        output_shape::NTuple{1,Int}

        output::Matrix{Float32}
        δ::Array{Float32,N}

        function Flatten(;input_shape::Tuple)
            output_shape = (prod(input_shape),)

            new{length(input_shape) + 1}(input_shape, output_shape)
        end
    end

    function initialize!(layer::Flatten, mini_batch::Int)
        isdefined(layer, :output) && size(layer.output, ndims(layer.output)) == mini_batch && return
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate!(layer::Flatten, input::Array{Float32})
        length(layer.output) == length(input) || throw(DimensionMismatch("unexpected reshape input size"))
        copyto!(layer.output, input)
    end

    supports_backward(::Flatten) = true

    function backward!(layer::Flatten, input, δₗ₊₁)
        length(layer.δ) == length(δₗ₊₁) || throw(DimensionMismatch("unexpected reshape gradient size"))
        copyto!(layer.δ, δₗ₊₁)
        return layer.δ
    end

    function update!(layer::Flatten, optimizer::String, input, delta, α::Float64, parameters::Tuple, direction::Int=1)
        backward!(layer, input, delta)
    end

    function save_layer(layer::Flatten, file, id::String)
        write(file, id, "Flatten")
        write(file, id*"input_shape", collect(layer.input_shape))
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        return (input_shape=input_shape,)
    end
end
