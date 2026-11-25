"""Channel-wise batch normalization for width × height × channels × batch tensors."""
struct BatchNorm2D{T<:AbstractFloat,A<:AbstractFloat,V<:AbstractVector{T},S<:AbstractVector{A}} <: AbstractLayer
    scale::V
    bias::V
    running_mean::S
    running_variance::S
    momentum::A
    epsilon::A
end

function BatchNorm2D(channels::Int; momentum::Real=1//10, epsilon::Real=1//100000,
                     precision::Type{T}=Float32) where {T<:AbstractFloat}
    checked_precision(T)
    A = accumulation_type(T)
    μ, ε = A(momentum), A(epsilon)
    channels > 0 && 0 <= μ <= 1 && 0 < ε < Inf ||
        throw(ArgumentError("BatchNorm2D requires positive channels and epsilon and momentum in [0, 1]"))
    return BatchNorm2D(ones(T, channels), zeros(T, channels), zeros(A, channels),
                       ones(A, channels), μ, ε)
end

precision_type(::BatchNorm2D{T}) where {T} = T
input_shape(layer::BatchNorm2D) = (nothing, nothing, length(layer.scale))
output_shape(::BatchNorm2D, input) = spatial_shape(input)
layer_parameters(layer::BatchNorm2D) = (; scale=layer.scale, bias=layer.bias)
persistent_buffers(layer::BatchNorm2D) = (; mean=layer.running_mean, variance=layer.running_variance)

mutable struct BatchNormCache{T,A,V<:AbstractVector{A},W<:AbstractArray{A,4}}
    scale_gradient::V
    bias_gradient::V
    mean::W
    variance::W
    inverse_std::W
    training::Bool
    normalized::W
    product::W
    output::AbstractArray{T,4}
    input_gradient::AbstractArray{T,4}

    function BatchNormCache(::Type{T}, scale::V, bias::V, mean::W) where
            {T,A,V<:AbstractVector{A},W<:AbstractArray{A,4}}
        return new{T,A,V,W}(scale, bias, mean, similar(mean), similar(mean), false)
    end
end

channel_axes(v) = reshape(v, 1, 1, length(v), 1)

function layer_cache(layer::BatchNorm2D{T}) where {T}
    return BatchNormCache(T, similar(layer.running_mean), similar(layer.running_mean),
                          similar(channel_axes(layer.running_mean)))
end

layer_gradients(::BatchNorm2D, cache::BatchNormCache) =
    (; scale=cache.scale_gradient, bias=cache.bias_gradient)

function forward!(layer::BatchNorm2D{T,A}, cache::BatchNormCache, input::AbstractArray{T,4}, mode) where {T,A}
    size(input, 3) == length(layer.scale) || throw(DimensionMismatch("BatchNorm2D channel count differs"))
    isempty(input) && throw(ArgumentError("BatchNorm2D requires nonempty input"))
    if !isdefined(cache, :output) || size(cache.output) != size(input)
        cache.normalized = similar(input, A)
        cache.product = similar(cache.normalized)
        cache.output = similar(input)
        cache.input_gradient = similar(input)
    end
    z, μ, σ² = cache.normalized, cache.mean, cache.variance
    cache.training = mode isa TrainingMode
    gpu_array(input) && return gpu_batch_norm_forward!(layer, cache, input)
    if cache.training
        count = length(input) ÷ size(input, 3)
        n = A(count)
        reduce_sum!(A, μ, input)
        broadcast_tensor!(/, μ, μ, n)
        broadcast_tensor!((x, mean) -> A(x) - mean, z, input, μ)
        reduce_sum!(abs2, σ², z)
        broadcast_tensor!(/, σ², σ², n)
        weight = layer.momentum
        correction = count > 1 ? n / A(count - 1) : one(A)
        update(previous, current, rate, scale) = (1 - rate) * previous + scale * current
        broadcast_tensor!(update, layer.running_mean, layer.running_mean, vec(μ), weight, weight)
        broadcast_tensor!(update, layer.running_variance, layer.running_variance, vec(σ²),
                          weight, weight * correction)
    else
        copy_tensor!(μ, channel_axes(layer.running_mean))
        copy_tensor!(σ², channel_axes(layer.running_variance))
        broadcast_tensor!((x, mean) -> A(x) - mean, z, input, μ)
    end
    broadcast_tensor!((variance, epsilon) -> inv(sqrt(variance + epsilon)), cache.inverse_std, σ², layer.epsilon)
    broadcast_tensor!(*, z, z, cache.inverse_std)
    broadcast_tensor!((x, scale, bias) -> x * scale + bias, cache.output,
                      z, channel_axes(layer.scale), channel_axes(layer.bias))
    return cache.output
end

function backward!(layer::BatchNorm2D{T,A}, cache::BatchNormCache, input, ∇y;
                   compute_parameters=true) where {T,A}
    z, ∇x = cache.normalized, cache.input_gradient
    ∑∇y, ∑∇yz = cache.mean, cache.variance
    if cache.training || compute_parameters
        reduce_sum!(A, ∑∇y, ∇y)
        broadcast_tensor!((dy, x) -> A(dy) * x, cache.product, ∇y, z)
        reduce_sum!(identity, ∑∇yz, cache.product)
    end
    γ = channel_axes(layer.scale)
    if cache.training
        n = A(length(input) ÷ size(input, 3))
        input_gradient(scale, inv_std, dy, s, x, p, n) = scale * inv_std * (A(dy) - (s + x * p) / n)
        broadcast_tensor!(input_gradient, ∇x, γ, cache.inverse_std, ∇y, ∑∇y, z, ∑∇yz, n)
    else
        broadcast_tensor!((scale, inv_std, dy) -> scale * inv_std * A(dy), ∇x, γ, cache.inverse_std, ∇y)
    end
    if compute_parameters
        batch = A(size(input, 4))
        if gpu_array(input)
            launch!(batch_norm_parameter_gradients_gpu!, cache.scale_gradient, cache.bias_gradient,
                    ∑∇yz, ∑∇y, batch)
        else
            broadcast_tensor!(/, cache.scale_gradient, vec(∑∇yz), batch)
            broadcast_tensor!(/, cache.bias_gradient, vec(∑∇y), batch)
        end
    end
    return ∇x
end
