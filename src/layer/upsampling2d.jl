module UpSampling2DM
    using LoopVectorization

    mutable struct UpSampling2D
        save_layer::Any
        load_layer::Any
        activate::Any
        initialize::Any
        update::Any

        input_shape::Tuple
        output_shape::Tuple
        size::Tuple{Int64, Int64}
        activation_function::Module

        value::Array{Float32}
        output::Array{Float32}
        ∇biases::Array{Float32}
        δ::Array{Float32}

        function UpSampling2D(;input_shape::Tuple, size::Tuple{Int64, Int64}, activation_function::Module)
            output_shape = (input_shape[1]*size[1], input_shape[2]*size[2], input_shape[3])
            new(save_UpSampling2D, load_UpSampling2D, activate_UpSampling2D, init_UpSampling2D, update_UpSampling2D, input_shape, output_shape, size, activation_function)
        end
    end

    function init_UpSampling2D(layer::UpSampling2D, mini_batch::Int64)
        layer.value = zeros(Float32, layer.output_shape..., mini_batch)
        layer.output = zeros(Float32, layer.output_shape..., mini_batch)
        layer.∇biases = zeros(Float32, layer.output_shape..., mini_batch)
        layer.δ = zeros(Float32, layer.input_shape..., mini_batch)
    end

    function activate_UpSampling2D(layer::UpSampling2D, input::Array{Float32})
        x, y = layer.size
        @avxt for i in axes(input, 1), j in axes(input, 2), c in axes(input, 3), b in axes(input, 4)
            for s₁ in 1:layer.size[1], s₂ in 1:layer.size[2]
                layer.value[(i-1)*x+s₁, (j-1)*y+s₂, c, b] = input[i, j, c, b]
            end
        end

        layer.activation_function.func!(layer.output, layer.value)
    end

    function update_UpSampling2D(layer::UpSampling2D, optimizer::String, input::Array{Float32}, δₗ₊₁::Array{Float32}, α::Float64, parameters::Tuple, direction::Int64=0)
        layer.activation_function.get_∇biases!(layer.∇biases, layer.value, δₗ₊₁)
        layer.δ .= 0.0f0
        
        x, y = layer.size
        @avxt for i in axes(layer.δ, 1), j in axes(layer.δ, 2), c in axes(layer.δ, 3), b in axes(layer.δ, 4)
            for s₁ in 1:layer.size[1], s₂ in 1:layer.size[2]
                layer.δ[i, j, c, b] += layer.∇biases[(i-1)*x+s₁, (j-1)*y+s₂, c, b]
            end
        end
    end

    function save_UpSampling2D(layer::UpSampling2D, file::Any, id::Int64)
        write(file, string(id), "UpSampling2D")
        write(file, string(id)*"input_shape", collect(layer.input_shape))
        write(file, string(id)*"size", collect(layer.size))
        write(file, string(id)*"activation_function", layer.activation_function.get_name())
    end

    function load_UpSampling2D(layer::UpSampling2D, file::Any, id::Int64)
    end

    function get_args(file::Any, id::Int64)
        input_shape = tuple(read(file, string(id)*"input_shape")...)
        size = tuple(read(file, string(id)*"size")...)
        return (input_shape=input_shape, size=size)
    end
end
