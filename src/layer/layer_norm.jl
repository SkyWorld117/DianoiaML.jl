"""Normalize the first axis with a learned scale and bias per feature."""
struct LayerNorm{T<:AbstractFloat,A<:AbstractFloat} <: AbstractLayer
    scale::Vector{T}
    bias::Vector{T}
    epsilon::A
end

function LayerNorm(width::Int; epsilon::Real=1//100000, precision::Type{T}=Float32) where {T<:AbstractFloat}
    checked_precision(T)
    width > 0 || throw(ArgumentError("LayerNorm requires positive features"))
    A = accumulation_type(T)
    ε = convert(A, epsilon)
    0 < ε < Inf || throw(ArgumentError("LayerNorm epsilon must be positive and finite"))
    return LayerNorm(ones(T, width), zeros(T, width), ε)
end

precision_type(::LayerNorm{T}) where {T} = T
input_shape(layer::LayerNorm) = (length(layer.scale), Colon())
output_shape(layer::LayerNorm, input) = feature_shape(input, length(layer.scale))
layer_parameters(layer::LayerNorm) = (; scale=layer.scale, bias=layer.bias)

mutable struct LayerNormCache{T,A}
    scale_gradient::Vector{A}
    bias_gradient::Vector{A}
    normalized::Array{A}
    inverse_std::Vector{A}
    output::Array{T}
    input_gradient::Array{T}
    LayerNormCache{T,A}(scale, bias) where {T,A} = new{T,A}(scale, bias)
end

layer_cache(layer::LayerNorm{T,A}) where {T,A} =
    LayerNormCache{T,A}(similar(layer.scale, A), similar(layer.bias, A))
layer_gradients(::LayerNorm, cache::LayerNormCache) = (; scale=cache.scale_gradient, bias=cache.bias_gradient)

function resize_norm!(cache::LayerNormCache{T,A}, dims) where {T,A}
    isdefined(cache, :output) && size(cache.output) == dims && return
    cache.normalized = Array{A}(undef, dims)
    cache.inverse_std = Vector{A}(undef, prod(Base.tail(dims)))
    cache.output = Array{T}(undef, dims)
    cache.input_gradient = similar(cache.output)
    return
end

function forward!(layer::LayerNorm{T,A}, cache::LayerNormCache{T,A}, input::Array{T}, mode) where {T,A}
    width = length(layer.scale)
    size(input, 1) == width || throw(DimensionMismatch("LayerNorm feature count differs"))
    resize_norm!(cache, size(input))
    x = reshape(input, width, :)
    z = reshape(cache.normalized, size(x))
    y = reshape(cache.output, size(x))
    γ, β, ε = layer.scale, layer.bias, layer.epsilon
    count = A(width)
    @inbounds for col in axes(x, 2)
        total = zero(A)
        @simd for i in 1:width
            total += convert(A, x[i, col])
        end
        μ = total / count
        squared = zero(A)
        @simd for i in 1:width
            centered = convert(A, x[i, col]) - μ
            z[i, col] = centered
            squared += abs2(centered)
        end
        σ² = squared / count
        σ⁻¹ = inv(sqrt(σ² + ε))
        cache.inverse_std[col] = σ⁻¹
        @simd for i in 1:width
            z[i, col] *= σ⁻¹
            y[i, col] = z[i, col] * convert(A, γ[i]) + convert(A, β[i])
        end
    end
    return cache.output
end

function backward!(layer::LayerNorm{T,A}, cache::LayerNormCache{T,A}, input, upstream; compute_parameters=true) where {T,A}
    size(upstream) == size(input) == size(cache.output) ||
        throw(DimensionMismatch("LayerNorm cotangent shape differs"))
    width = length(layer.scale)
    ∇x = reshape(cache.input_gradient, width, :)
    z = reshape(cache.normalized, size(∇x))
    ∇y = reshape(upstream, size(∇x))
    γ = layer.scale
    ∇γ, ∇β = cache.scale_gradient, cache.bias_gradient
    count = A(width)
    if compute_parameters
        fill!(∇γ, zero(A))
        fill!(∇β, zero(A))
    end
    @inbounds for col in axes(∇y, 2)
        total, projection = zero(A), zero(A)
        @simd for i in 1:width
            h = convert(A, ∇y[i, col]) * convert(A, γ[i])
            total += h
            projection += h * z[i, col]
        end
        μ, mean_projection = total / count, projection / count
        σ⁻¹ = cache.inverse_std[col]
        @simd for i in 1:width
            h = convert(A, ∇y[i, col]) * convert(A, γ[i])
            ∇x[i, col] = σ⁻¹ * (h - μ - z[i, col] * mean_projection)
        end
        if compute_parameters
            @simd for i in 1:width
                g = convert(A, ∇y[i, col])
                ∇γ[i] += g * z[i, col]
                ∇β[i] += g
            end
        end
    end
    if compute_parameters
        batch = A(size(input, ndims(input)))
        ∇γ ./= batch
        ∇β ./= batch
    end
    return cache.input_gradient
end
