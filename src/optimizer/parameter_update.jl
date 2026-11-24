abstract type AbstractOptimizer end

"""Stochastic gradient descent with optional classical momentum."""
struct SGD{T<:AbstractFloat} <: AbstractOptimizer
    lr::T
    momentum::T
    function SGD(; lr::Real=1//100, momentum::Real=zero(lr))
        rate, coefficient = promote(float(lr), float(momentum))
        validate_rate(rate)
        0 <= coefficient < 1 || throw(ArgumentError("momentum must be in [0, 1)"))
        new{typeof(rate)}(rate, coefficient)
    end
end

struct Adam{T<:AbstractFloat} <: AbstractOptimizer
    lr::T
    beta1::T
    beta2::T
    epsilon::T
    function Adam(; lr::Real=1//1000, beta1::Real=9//10, beta2::Real=999//1000, epsilon::Real=1//100000000)
        rate, b1, b2, eps = adaptive_settings(lr, beta1, beta2, epsilon)
        new{typeof(rate)}(rate, b1, b2, eps)
    end
end

"""Adam with decoupled weight decay applied at the parameter's learning rate."""
struct AdamW{T<:AbstractFloat} <: AbstractOptimizer
    lr::T
    beta1::T
    beta2::T
    epsilon::T
    weight_decay::T
    function AdamW(; lr::Real=1//1000, beta1::Real=9//10, beta2::Real=999//1000,
                   epsilon::Real=1//100000000, weight_decay::Real=1//100)
        rate, b1, b2, eps, decay = adaptive_settings(lr, beta1, beta2, epsilon, weight_decay)
        0 <= decay < Inf || throw(ArgumentError("weight decay must be nonnegative and finite"))
        new{typeof(rate)}(rate, b1, b2, eps, decay)
    end
end

"""Uncentered RMSProp with a running average of squared gradients."""
struct RMSProp{T<:AbstractFloat} <: AbstractOptimizer
    lr::T
    rho::T
    epsilon::T
    function RMSProp(; lr::Real=1//100, rho::Real=99//100, epsilon::Real=1//100000000)
        rate, decay, eps = optimizer_settings(lr, rho, epsilon)
        validate_rate(rate)
        0 <= decay < 1 || throw(ArgumentError("rho must be in [0, 1)"))
        validate_epsilon(eps)
        new{typeof(rate)}(rate, decay, eps)
    end
end

struct AdaBelief{T<:AbstractFloat} <: AbstractOptimizer
    lr::T
    beta1::T
    beta2::T
    epsilon::T
    function AdaBelief(; lr::Real=1//1000, beta1::Real=9//10, beta2::Real=999//1000, epsilon::Real=1//100000000)
        rate, b1, b2, eps = adaptive_settings(lr, beta1, beta2, epsilon)
        new{typeof(rate)}(rate, b1, b2, eps)
    end
end

function validate_rate(rate)
    0 <= rate < Inf || throw(ArgumentError("learning rate must be nonnegative and finite"))
end

function optimizer_settings(values...)
    A = accumulation_type(promote_type(map(x -> typeof(float(x)), values)...))
    return map(x -> convert(A, x), values)
end

function validate_epsilon(epsilon)
    0 < epsilon < Inf || throw(ArgumentError("optimizer epsilon must be positive and finite"))
end

function adaptive_settings(lr, beta1, beta2, epsilon, extra...)
    values = optimizer_settings(lr, beta1, beta2, epsilon, extra...)
    rate, b1, b2, eps = values
    validate_rate(rate)
    0 <= b1 < 1 && 0 <= b2 < 1 || throw(ArgumentError("optimizer decays must be in [0, 1)"))
    validate_epsilon(eps)
    return values
end

moment_count(optimizer::SGD) = iszero(optimizer.momentum) ? 0 : 1
moment_count(::RMSProp) = 1
moment_count(::Union{Adam,AdamW,AdaBelief}) = 2

function update_parameters!(::SGD, weights, gradient, ::Nothing, ::Nothing, rate, iteration)
    weights .-= eltype(gradient)(rate) .* gradient
    return
end

function update_parameters!(optimizer::SGD, weights, gradient, velocity::Array, ::Nothing, rate, iteration)
    A = eltype(gradient)
    step, momentum = A(rate), A(optimizer.momentum)
    @inbounds @simd for i in eachindex(weights, gradient, velocity)
        v = momentum * velocity[i] + gradient[i]
        velocity[i] = v
        weights[i] -= step * v
    end
    return
end

function update_parameters!(optimizer::RMSProp, weights, gradient, squares, ::Nothing, rate, iteration)
    A = eltype(gradient)
    step, epsilon = A(rate), A(optimizer.epsilon)
    decay, gain = A(optimizer.rho), A(1 - optimizer.rho)
    @inbounds @simd for i in eachindex(weights, gradient, squares)
        g = gradient[i]
        v = decay * squares[i] + gain * abs2(g)
        squares[i] = v
        weights[i] -= step * g / (sqrt(v) + epsilon)
    end
    return
end

weight_scale(::Union{Adam,AdaBelief}, rate, ::Type{A}) where {A} = one(A)
weight_scale(optimizer::AdamW, rate, ::Type{A}) where {A} = one(A) - A(rate * optimizer.weight_decay)

function update_parameters!(optimizer::Union{Adam,AdamW,AdaBelief}, weights, gradient, first, second, rate, iteration)
    A = eltype(gradient)
    P = promote_type(A, typeof(optimizer.beta1))
    b1, b2 = P(optimizer.beta1), P(optimizer.beta2)
    decay1, decay2 = A(b1), A(b2)
    gain1, gain2 = A(1 - b1), A(1 - b2)
    step = A(rate) / A(1 - b1^iteration)
    correction = A(inv(1 - b2^iteration))
    epsilon = A(optimizer.epsilon)
    scale = weight_scale(optimizer, rate, A)
    belief = optimizer isa AdaBelief
    @inbounds @simd for i in eachindex(weights, gradient, first, second)
        g = gradient[i]
        m = decay1 * first[i] + gain1 * g
        deviation = ifelse(belief, g - m, g)
        v = decay2 * second[i] + gain2 * deviation^2
        first[i], second[i] = m, v
        weights[i] = scale * weights[i] - step * m / (sqrt(v * correction) + epsilon)
    end
    return
end
