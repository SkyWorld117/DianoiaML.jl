using NNlib: softmax!

abstract type AbstractActivation <: AbstractLayer end

struct Identity <: AbstractActivation end
struct ReLU <: AbstractActivation end
struct Sigmoid <: AbstractActivation end
struct Softmax <: AbstractActivation end
struct Tanh <: AbstractActivation end
struct GELU <: AbstractActivation end

output_shape(::AbstractActivation, input) = input

activate_values!(out, ::Identity, x) = copyto!(out, x)
activate_values!(out, ::ReLU, x) = (out .= max.(zero(eltype(x)), x))
activate_values!(out, ::Tanh, x) = (out .= tanh.(x))

sigmoid(x) = x >= 0 ? inv(one(x) + exp(-x)) : exp(x) / (one(x) + exp(x))
activate_values!(out, ::Sigmoid, x) = (out .= sigmoid.(x))

gelu_tanh(x, c, k) = tanh(c * (x + k * x^3))
function gelu(x, c, k)
    x == -Inf && return zero(x)
    return x * ((one(x) + gelu_tanh(x, c, k)) / 2)
end

function gelu_derivative(x, c, k)
    t = gelu_tanh(x, c, k)
    abs(t) == 1 && return (one(t) + t) / 2
    return (one(t) + t + x * (one(t) - t^2) * c * (one(x) + 3 * k * x^2)) / 2
end

gelu_constants(::Type{A}) where {A} = sqrt(A(2) / A(π)), A(44715//1000000)

const PointwiseActivation = Union{Identity,ReLU,Sigmoid,Tanh,GELU}

pointwise_arguments(::PointwiseActivation, ::Type) = ()
pointwise_arguments(::GELU, ::Type{T}) where {T} = gelu_constants(accumulation_type(T))

@inline activate_scalar(::Identity, x, args) = x
@inline activate_scalar(::ReLU, x, args) = max(zero(x), x)
@inline activate_scalar(::Sigmoid, x, args) = sigmoid(x)
@inline activate_scalar(::Tanh, x, args) = tanh(x)
@inline activate_scalar(::GELU, x, args) = gelu(typeof(first(args))(x), args...)

function activate_values!(out, ::GELU, x)
    A = accumulation_type(eltype(x))
    c, k = gelu_constants(A)
    out .= gelu.(A.(x), c, k)
end

function activate_values!(out, ::Softmax, x)
    gpu_array(x) && return gpu_softmax!(out, x)
    eltype(x) <: Union{Float32,Float64} && return softmax!(out, x; dims=1)
    A = accumulation_type(eltype(x))
    width = size(x, 1)
    input, output = reshape(x, width, :), reshape(out, width, :)
    @inbounds for column in axes(input, 2)
        peak = A(maximum(view(input, :, column)))
        total = sum(i -> exp(A(input[i, column]) - peak), axes(input, 1))
        @simd for i in axes(input, 1)
            output[i, column] = exp(A(input[i, column]) - peak) / total
        end
    end
    return out
end

activation_gradient!(∇x, ::Identity, x, y, ∇y) = copyto!(∇x, ∇y)
activation_gradient!(∇x, ::ReLU, x, y, ∇y) = (∇x .= ifelse.(x .> 0, ∇y, zero(eltype(∇y))))
activation_gradient!(∇x, ::Tanh, x, y, ∇y) = (∇x .= ∇y .* (1 .- abs2.(y)))
activation_gradient!(∇x, ::Sigmoid, x, y, ∇y) = (∇x .= ∇y .* y .* (1 .- y))

function activation_gradient!(∇x, ::GELU, x, y, ∇y)
    A = accumulation_type(eltype(x))
    c, k = gelu_constants(A)
    ∇x .= gelu_derivative.(A.(x), c, k) .* A.(∇y)
end

function activation_gradient!(out, ::Softmax, x, y, upstream)
    gpu_array(y) && return gpu_softmax_backward!(out, y, upstream)
    A = accumulation_type(eltype(y))
    width = size(y, 1)
    ∇x, p, ∇p = map(a -> reshape(a, width, :), (out, y, upstream))
    @inbounds for column in axes(p, 2)
        projection = zero(A)
        @simd for i in axes(p, 1)
            projection += A(p[i, column]) * A(∇p[i, column])
        end
        @simd for i in axes(p, 1)
            ∇x[i, column] = p[i, column] * (∇p[i, column] - projection)
        end
    end
    return out
end

mutable struct ActivationCache
    output::AbstractArray
    gradient::AbstractArray
    ActivationCache() = new()
end

layer_cache(::AbstractActivation) = ActivationCache()

function forward!(layer::AbstractActivation, cache::ActivationCache, input, mode)
    if !isdefined(cache, :output) || size(cache.output) != size(input) ||
       eltype(cache.output) !== eltype(input) || !same_device(cache.output, input)
        cache.output, cache.gradient = similar(input), similar(input)
    end
    return activate_values!(cache.output, layer, input)
end

backward!(layer::AbstractActivation, cache::ActivationCache, input, delta; compute_parameters=true) =
    activation_gradient!(cache.gradient, layer, input, cache.output, delta)
