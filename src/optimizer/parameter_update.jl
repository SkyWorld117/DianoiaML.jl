function update_parameters!(optimizer, weights, gradient, first, second, rate, parameters)
    A = eltype(gradient)
    rate = A(rate)
    if optimizer == "SGD" || optimizer == "Minibatch_GD"
        weights .-= rate .* gradient
        return
    end

    belief = optimizer == "AdaBelief"
    (belief || optimizer == "Adam") || throw(ArgumentError("unknown optimizer $optimizer"))
    t, β₁, β₂, ϵ = parameters
    # Preserve the configuration precision for bias correction, then update
    # the moments in the gradient's accumulation type.
    P = promote_type(A, typeof(β₁), typeof(β₂))
    β₁p, β₂p = P(β₁), P(β₂)
    decay1, decay2 = A(β₁p), A(β₂p)
    gain1, gain2 = A(1 - β₁p), A(1 - β₂p)
    step = rate / A(1 - β₁p^t)
    correction = A(inv(1 - β₂p^t))
    epsilon = A(ϵ)
    @inbounds @simd for i in eachindex(weights, gradient, first, second)
        g = gradient[i]
        m = decay1 * first[i] + gain1 * g
        deviation = ifelse(belief, g - m, g)
        v = decay2 * second[i] + gain2 * deviation^2
        first[i] = m
        second[i] = v
        weights[i] -= step * m / (sqrt(v * correction) + epsilon)
    end
end
