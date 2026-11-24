abstract type AbstractOptimizer end

struct SGD{T<:AbstractFloat} <: AbstractOptimizer
    lr::T
    function SGD(; lr::Real=1//100)
        validate_rate(lr)
        rate = float(lr)
        new{typeof(rate)}(rate)
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

function adaptive_settings(lr, beta1, beta2, epsilon)
    values = (lr, beta1, beta2, epsilon)
    A = accumulation_type(promote_type(map(x -> typeof(float(x)), values)...))
    rate, b1, b2, eps = map(x -> convert(A, x), values)
    validate_rate(rate)
    0 <= b1 < 1 && 0 <= b2 < 1 || throw(ArgumentError("optimizer decays must be in [0, 1)"))
    0 < eps < Inf || throw(ArgumentError("optimizer epsilon must be positive and finite"))
    return rate, b1, b2, eps
end

has_moments(::SGD) = false
has_moments(::Union{Adam,AdaBelief}) = true

function update_parameters!(::SGD, weights, gradient, first, second, rate, iteration)
    weights .-= eltype(gradient)(rate) .* gradient
    return
end

function update_parameters!(optimizer::Union{Adam,AdaBelief}, weights, gradient, first, second, rate, iteration)
    A = eltype(gradient)
    P = promote_type(A, typeof(optimizer.beta1))
    b1, b2 = P(optimizer.beta1), P(optimizer.beta2)
    decay1, decay2 = A(b1), A(b2)
    gain1, gain2 = A(1 - b1), A(1 - b2)
    step = A(rate) / A(1 - b1^iteration)
    correction = A(inv(1 - b2^iteration))
    epsilon = A(optimizer.epsilon)
    belief = optimizer isa AdaBelief
    @inbounds @simd for i in eachindex(weights, gradient, first, second)
        g = gradient[i]
        m = decay1 * first[i] + gain1 * g
        deviation = ifelse(belief, g - m, g)
        v = decay2 * second[i] + gain2 * deviation^2
        first[i], second[i] = m, v
        weights[i] -= step * m / (sqrt(v * correction) + epsilon)
    end
    return
end
