module LayerNormM
    using ..DianoiaML: AbstractLayer, checked_precision, accumulation_type,
                       read_array, write_array, read_number, write_number, update_parameters!
    import ..DianoiaML: precision_type, initialize!, activate!, backward!, update!,
                        supports_backward, parameter_gradients, save_layer, load_layer!

    """
        LayerNorm(; input_shape, epsilon=1//100000, precision=Float32)

    Normalize each feature vector independently along the first axis, with
    a learned scale and bias per feature. Examples occupy the last axis.
    """
    mutable struct LayerNorm{T<:AbstractFloat,A<:AbstractFloat,N} <: AbstractLayer
        input_shape::Tuple
        output_shape::Tuple
        epsilon::A
        scale::Vector{T}
        bias::Vector{T}
        scale_gradient::Vector{A}
        bias_gradient::Vector{A}
        scale_moments::NTuple{2,Vector{A}}
        bias_moments::NTuple{2,Vector{A}}
        normalized::Array{A,N}
        inverse_std::Vector{A}
        output::Array{T,N}
        δ::Array{T,N}

        function LayerNorm(; input_shape::Tuple, epsilon::Real=1//100000,
                           precision::Type{T}=Float32) where {T<:AbstractFloat}
            checked_precision(T)
            !isempty(input_shape) && all(d -> d isa Int && d > 0, input_shape) ||
                throw(ArgumentError("LayerNorm requires positive integer input dimensions"))
            A = accumulation_type(T)
            eps = convert(A, epsilon)
            0 < eps < Inf || throw(ArgumentError("LayerNorm epsilon must be positive and finite"))
            width = first(input_shape)
            new{T,A,length(input_shape) + 1}(input_shape, input_shape, eps,
                ones(T, width), zeros(T, width), zeros(A, width), zeros(A, width),
                (zeros(A, width), zeros(A, width)), (zeros(A, width), zeros(A, width)))
        end
    end

    precision_type(::LayerNorm{T}) where {T} = T
    supports_backward(::LayerNorm) = true
    parameter_gradients(layer::LayerNorm) =
        (; scale=(layer.scale, layer.scale_gradient), bias=(layer.bias, layer.bias_gradient))

    initialize!(layer::LayerNorm, batch::Int) = resize_buffers!(layer, (layer.input_shape..., batch))

    function resize_buffers!(layer::LayerNorm{T,A}, dims) where {T,A}
        isdefined(layer, :output) && size(layer.output) == dims && return
        layer.normalized = Array{A}(undef, dims)
        layer.inverse_std = Vector{A}(undef, prod(Base.tail(dims)))
        layer.output = Array{T}(undef, dims)
        layer.δ = similar(layer.output)
        return
    end

    function activate!(layer::LayerNorm{T,A,N}, input::Array{T,N}) where {T,A,N}
        width = length(layer.scale)
        size(input, 1) == width || throw(DimensionMismatch("LayerNorm feature count differs"))
        resize_buffers!(layer, size(input))
        x = reshape(input, width, :)
        z = reshape(layer.normalized, size(x))
        output = reshape(layer.output, size(x))
        count = A(width)
        @inbounds for col in axes(x, 2)
            total = zero(A)
            @simd for i in 1:width
                total += convert(A, x[i, col])
            end
            mean = total / count
            squared = zero(A)
            @simd for i in 1:width
                centered = convert(A, x[i, col]) - mean
                z[i, col] = centered
                squared += abs2(centered)
            end
            inverse_std = inv(sqrt(squared / count + layer.epsilon))
            layer.inverse_std[col] = inverse_std
            @simd for i in 1:width
                z[i, col] *= inverse_std
                output[i, col] = z[i, col] * convert(A, layer.scale[i]) + convert(A, layer.bias[i])
            end
        end
        return layer.output
    end

    function backward!(layer::LayerNorm{T,A}, input, delta::Array{T}; compute_parameters=true) where {T,A}
        size(delta) == size(input) == size(layer.output) ||
            throw(DimensionMismatch("LayerNorm cotangent shape differs"))
        width = length(layer.scale)
        xgrad = reshape(layer.δ, width, :)
        z = reshape(layer.normalized, size(xgrad))
        dy = reshape(delta, size(xgrad))
        count = A(width)
        if compute_parameters
            fill!(layer.scale_gradient, zero(A))
            fill!(layer.bias_gradient, zero(A))
        end
        @inbounds for col in axes(dy, 2)
            total, projection = zero(A), zero(A)
            @simd for i in 1:width
                h = convert(A, dy[i, col]) * convert(A, layer.scale[i])
                total += h
                projection += h * z[i, col]
            end
            mean, mean_projection = total / count, projection / count
            inverse_std = layer.inverse_std[col]
            @simd for i in 1:width
                h = convert(A, dy[i, col]) * convert(A, layer.scale[i])
                xgrad[i, col] = inverse_std * (h - mean - z[i, col] * mean_projection)
            end
            if compute_parameters
                @simd for i in 1:width
                    g = convert(A, dy[i, col])
                    layer.scale_gradient[i] += g * z[i, col]
                    layer.bias_gradient[i] += g
                end
            end
        end
        if compute_parameters
            batch = A(size(input, ndims(input)))
            layer.scale_gradient ./= batch
            layer.bias_gradient ./= batch
        end
        return layer.δ
    end

    function update!(layer::LayerNorm, optimizer::String, input, delta, rate::Real, settings::Tuple, direction::Int=1)
        backward!(layer, input, delta; compute_parameters=direction != 0)
        direction == 0 && return layer.δ
        update_parameters!(optimizer, layer.scale, layer.scale_gradient, layer.scale_moments...,
                           rate * direction, settings)
        update_parameters!(optimizer, layer.bias, layer.bias_gradient, layer.bias_moments...,
                           rate * direction, settings)
        return layer.δ
    end

    function save_layer(layer::LayerNorm, file, id::String)
        write(file, id, "LayerNorm")
        write(file, id * "input_shape", collect(layer.input_shape))
        write_number(file, id * "epsilon", layer.epsilon)
        write_array(file, id * "scale", layer.scale)
        write_array(file, id * "bias", layer.bias)
    end

    function get_args(file, id::String)
        input_shape = Tuple(read(file, id * "input_shape"))
        epsilon = read_number(file, id * "epsilon")
        return (; input_shape, epsilon)
    end

    function load_layer!(layer::LayerNorm, file, id::String)
        copyto!(layer.scale, read_array(file, id * "scale"))
        copyto!(layer.bias, read_array(file, id * "bias"))
        return
    end
end
