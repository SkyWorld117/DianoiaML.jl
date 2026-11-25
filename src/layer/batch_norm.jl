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
    if cache.training
        count = length(input) ÷ size(input, 3)
        n = A(count)
        reduce_sum!(A, μ, input)
        μ ./= n
        z .= A.(input) .- μ
        reduce_sum!(abs2, σ², z)
        σ² ./= n
        weight = layer.momentum
        correction = count > 1 ? n / A(count - 1) : one(A)
        layer.running_mean .= (1 - weight) .* layer.running_mean .+ weight .* vec(μ)
        layer.running_variance .= (1 - weight) .* layer.running_variance .+ (weight * correction) .* vec(σ²)
    else
        copyto!(μ, channel_axes(layer.running_mean))
        copyto!(σ², channel_axes(layer.running_variance))
        z .= A.(input) .- μ
    end
    cache.inverse_std .= inv.(sqrt.(σ² .+ layer.epsilon))
    z .*= cache.inverse_std
    cache.output .= z .* channel_axes(layer.scale) .+ channel_axes(layer.bias)
    return cache.output
end

function backward!(layer::BatchNorm2D{T,A}, cache::BatchNormCache, input, ∇y;
                   compute_parameters=true) where {T,A}
    z, ∇x = cache.normalized, cache.input_gradient
    ∑∇y, ∑∇yz = cache.mean, cache.variance
    if cache.training || compute_parameters
        reduce_sum!(A, ∑∇y, ∇y)
        cache.product .= A.(∇y) .* z
        reduce_sum!(identity, ∑∇yz, cache.product)
    end
    γ = channel_axes(layer.scale)
    if cache.training
        n = A(length(input) ÷ size(input, 3))
        ∇x .= γ .* cache.inverse_std .* (A.(∇y) .- (∑∇y .+ z .* ∑∇yz) ./ n)
    else
        ∇x .= γ .* cache.inverse_std .* A.(∇y)
    end
    if compute_parameters
        batch = A(size(input, 4))
        cache.scale_gradient .= vec(∑∇yz) ./ batch
        cache.bias_gradient .= vec(∑∇y) ./ batch
    end
    return ∇x
end
