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
        α, β₁, β₂, ε = adaptive_settings(lr, beta1, beta2, epsilon)
        new{typeof(α)}(α, β₁, β₂, ε)
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
        α, β₁, β₂, ε, λ = adaptive_settings(lr, beta1, beta2, epsilon, weight_decay)
        0 <= λ < Inf || throw(ArgumentError("weight decay must be nonnegative and finite"))
        new{typeof(α)}(α, β₁, β₂, ε, λ)
    end
end

"""Uncentered RMSProp with a running average of squared gradients."""
struct RMSProp{T<:AbstractFloat} <: AbstractOptimizer
    lr::T
    rho::T
    epsilon::T
    function RMSProp(; lr::Real=1//100, rho::Real=99//100, epsilon::Real=1//100000000)
        α, ρ, ε = optimizer_settings(lr, rho, epsilon)
        validate_rate(α)
        0 <= ρ < 1 || throw(ArgumentError("rho must be in [0, 1)"))
        validate_epsilon(ε)
        new{typeof(α)}(α, ρ, ε)
    end
end

struct AdaBelief{T<:AbstractFloat} <: AbstractOptimizer
    lr::T
    beta1::T
    beta2::T
    epsilon::T
    function AdaBelief(; lr::Real=1//1000, beta1::Real=9//10, beta2::Real=999//1000, epsilon::Real=1//100000000)
        α, β₁, β₂, ε = adaptive_settings(lr, beta1, beta2, epsilon)
        new{typeof(α)}(α, β₁, β₂, ε)
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
    α, β₁, β₂, ε = values
    validate_rate(α)
    0 <= β₁ < 1 && 0 <= β₂ < 1 || throw(ArgumentError("optimizer decays must be in [0, 1)"))
    validate_epsilon(ε)
    return values
end

moment_count(optimizer::SGD) = iszero(optimizer.momentum) ? 0 : 1
moment_count(::RMSProp) = 1
moment_count(::Union{Adam,AdamW,AdaBelief}) = 2

function update_parameters!(::SGD, w, ∇w, ::Nothing, ::Nothing, α, t)
    w .-= eltype(∇w)(α) .* ∇w
    return
end

function update_parameters!(optimizer::SGD, w, ∇w, velocity::AbstractArray, ::Nothing, rate, t)
    A = eltype(∇w)
    α, μ = A(rate), A(optimizer.momentum)
    gpu_array(w) && return launch!(momentum_gpu!, w, ∇w, velocity, α, μ)
    @inbounds @simd for i in eachindex(w, ∇w, velocity)
        v = μ * velocity[i] + ∇w[i]
        velocity[i] = v
        w[i] -= α * v
    end
    return
end

function update_parameters!(optimizer::RMSProp, w, ∇w, squares, ::Nothing, rate, t)
    A = eltype(∇w)
    α, ε = A(rate), A(optimizer.epsilon)
    ρ, gain = A(optimizer.rho), A(1 - optimizer.rho)
    gpu_array(w) && return launch!(rmsprop_gpu!, w, ∇w, squares, α, ε, ρ, gain)
    @inbounds @simd for i in eachindex(w, ∇w, squares)
        g = ∇w[i]
        v = ρ * squares[i] + gain * abs2(g)
        squares[i] = v
        w[i] -= α * g / (sqrt(v) + ε)
    end
    return
end

weight_scale(::Union{Adam,AdaBelief}, α, ::Type{A}) where {A} = one(A)
weight_scale(optimizer::AdamW, α, ::Type{A}) where {A} = one(A) - A(α * optimizer.weight_decay)

function adaptive_coefficients(optimizer, ::Type{A}, α, t) where {A}
    P = promote_type(A, typeof(optimizer.beta1))
    β₁, β₂ = P(optimizer.beta1), P(optimizer.beta2)
    decay₁, decay₂ = A(β₁), A(β₂)
    gain₁, gain₂ = A(1 - β₁), A(1 - β₂)
    α̂ = A(α) / A(1 - β₁^t)
    correction = A(inv(1 - β₂^t))
    ε = A(optimizer.epsilon)
    scale = weight_scale(optimizer, α, A)
    return α̂, ε, scale, decay₁, decay₂, gain₁, gain₂, correction
end

function update_parameters!(optimizer::Union{Adam,AdamW,AdaBelief}, w, ∇w, first, second, α, t)
    coefficients = adaptive_coefficients(optimizer, eltype(∇w), α, t)
    belief = optimizer isa AdaBelief
    if gpu_array(w)
        return launch!(adaptive_optimizer_gpu!, w, ∇w, first, second, coefficients..., belief)
    end
    α̂, ε, scale, decay₁, decay₂, gain₁, gain₂, correction = coefficients
    @inbounds @simd for i in eachindex(w, ∇w, first, second)
        g = ∇w[i]
        m = decay₁ * first[i] + gain₁ * g
        deviation = ifelse(belief, g - m, g)
        v = decay₂ * second[i] + gain₂ * deviation^2
        first[i], second[i] = m, v
        v̂ = v * correction
        w[i] = scale * w[i] - α̂ * m / (sqrt(v̂) + ε)
    end
    return
end

struct AdamWGroup{S,B}
    slots::S
    buffers::B
    width::Int
end
