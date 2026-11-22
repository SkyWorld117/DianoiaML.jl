module UpSampling2DM
    import ..DianoiaML: backward!, supports_backward
    import ..DianoiaML: AbstractLayer, initialize!, activate!, update!, save_layer
    using ..DianoiaML: activation_gradient!

    mutable struct UpSampling2D <: AbstractLayer
        input_shape::NTuple{3,Int}
        output_shape::NTuple{3,Int}
        size::NTuple{2,Int}
        activation_function::Module

        value::Array{Float32,4}
        output::Array{Float32,4}
        ∇biases::Array{Float32,4}
        δ::Array{Float32,4}

        function UpSampling2D(;input_shape::Tuple, size::NTuple{2,Int}, activation_function::Module)
            length(input_shape) == 3 && all(>(0), input_shape) && all(>(0), size) ||
                throw(ArgumentError("invalid upsampling shape or scale"))
            output_shape = (input_shape[1]*size[1], input_shape[2]*size[2], input_shape[3])
            new(input_shape, output_shape, size, activation_function)
        end
    end

    function initialize!(layer::UpSampling2D, mini_batch::Int)
        isdefined(layer, :value) && size(layer.value, 4) == mini_batch && return
        layer.value = zeros(Float32, layer.output_shape..., mini_batch)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.∇biases = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate!(layer::UpSampling2D, input::Array{Float32,4})
        size(input) == size(layer.δ) || throw(DimensionMismatch("unexpected upsampling input size"))
        x, y = layer.size
        @inbounds for b in axes(input, 4), c in axes(input, 3), j in axes(input, 2), sy in 1:y, i in axes(input, 1)
            @simd for sx in 1:x
                layer.value[(i - 1) * x + sx, (j - 1) * y + sy, c, b] = input[i, j, c, b]
            end
        end
        layer.activation_function.func!(layer.output, layer.value)
    end

    supports_backward(::UpSampling2D) = true

    function backward!(layer::UpSampling2D, input, δₗ₊₁)
        activation_gradient!(layer.∇biases, layer.activation_function, layer.value, δₗ₊₁)
        x, y = layer.size
        @inbounds for b in axes(layer.δ, 4), c in axes(layer.δ, 3), j in axes(layer.δ, 2), i in axes(layer.δ, 1)
            total = 0f0
            for sy in 1:y
                @simd for sx in 1:x
                    total += layer.∇biases[(i - 1) * x + sx, (j - 1) * y + sy, c, b]
                end
            end
            layer.δ[i, j, c, b] = total
        end
        return layer.δ
    end

    function update!(layer::UpSampling2D, optimizer::String, input, delta, α::Float64, parameters::Tuple, direction::Int=1)
        backward!(layer, input, delta)
    end

    function save_layer(layer::UpSampling2D, file, id::String)
        write(file, id, "UpSampling2D")
        write(file, id*"input_shape", collect(layer.input_shape))
        write(file, id*"size", collect(layer.size))
        write(file, id*"activation_function", layer.activation_function.get_name())
    end

    function get_args(file, id::String)
        input_shape = tuple(read(file, id*"input_shape")...)
        size = tuple(read(file, id*"size")...)
        return (input_shape=input_shape, size=size)
    end
end
