module ConstructiveM
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer
    mutable struct Constructive{N,M} <: AbstractLayer
        input_shape::Tuple
        output_shape::Tuple

        output::Array{Float32,M}
        δ::Array{Float32,N}

        function Constructive(;input_shape::Tuple, shape::Tuple)
            prod(input_shape) == prod(shape) || throw(DimensionMismatch("reshape changes the feature count"))
            new{length(input_shape) + 1, length(shape) + 1}(input_shape, shape)
        end
    end

    function initialize!(layer::Constructive, mini_batch::Int)
        isdefined(layer, :output) && size(layer.output, ndims(layer.output)) == mini_batch && return
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate!(layer::Constructive, input::Array{Float32})
        length(layer.output) == length(input) || throw(DimensionMismatch("unexpected reshape input size"))
        copyto!(layer.output, input)
    end

    function update!(layer::Constructive, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int=1)
        length(layer.δ) == length(δₗ₊₁) || throw(DimensionMismatch("unexpected reshape gradient size"))
        copyto!(layer.δ, δₗ₊₁)
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
