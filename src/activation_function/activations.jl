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

gelu_constants(::Type{A}) where {A} = sqrt(A(2) / A(pi)), A(44715//1000000)

function activate_values!(out, ::GELU, x)
    A = accumulation_type(eltype(x))
    c, k = gelu_constants(A)
    out .= gelu.(A.(x), c, k)
end

function activate_values!(out, ::Softmax, x)
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

activation_gradient!(out, ::Identity, x, y, dy) = copyto!(out, dy)
activation_gradient!(out, ::ReLU, x, y, dy) = (out .= ifelse.(x .> 0, dy, zero(eltype(dy))))
activation_gradient!(out, ::Tanh, x, y, dy) = (out .= dy .* (1 .- abs2.(y)))
activation_gradient!(out, ::Sigmoid, x, y, dy) = (out .= dy .* y .* (1 .- y))

function activation_gradient!(out, ::GELU, x, y, dy)
    A = accumulation_type(eltype(x))
    c, k = gelu_constants(A)
    out .= gelu_derivative.(A.(x), c, k) .* A.(dy)
end

function activation_gradient!(out, ::Softmax, x, y, dy)
    A = accumulation_type(eltype(y))
    width = size(y, 1)
    dst, prob, delta = map(a -> reshape(a, width, :), (out, y, dy))
    @inbounds for column in axes(prob, 2)
        projection = zero(A)
        @simd for i in axes(prob, 1)
            projection += A(prob[i, column]) * A(delta[i, column])
        end
        @simd for i in axes(prob, 1)
            dst[i, column] = prob[i, column] * (delta[i, column] - projection)
        end
    end
    return out
end

mutable struct ActivationCache
    output::Array
    gradient::Array
    ActivationCache() = new()
end

layer_cache(::AbstractActivation) = ActivationCache()

function forward!(layer::AbstractActivation, cache::ActivationCache, input, mode)
    if !isdefined(cache, :output) || size(cache.output) != size(input) || eltype(cache.output) !== eltype(input)
        cache.output, cache.gradient = similar(input), similar(input)
    end
    return activate_values!(cache.output, layer, input)
end

backward!(layer::AbstractActivation, cache::ActivationCache, input, delta; compute_parameters=true) =
    activation_gradient!(cache.gradient, layer, input, cache.output, delta)
