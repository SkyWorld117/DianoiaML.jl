function update_parameters!(optimizer, weights, gradient, first, second, rate, parameters)
    if optimizer == "SGD" || optimizer == "Minibatch_GD"
        weights .-= rate .* gradient
        return
    end

    belief = optimizer == "AdaBelief"
    (belief || optimizer == "Adam") || throw(ArgumentError("unknown optimizer $optimizer"))
    t, β₁, β₂, ϵ = parameters
    decay1, decay2 = Float32(β₁), Float32(β₂)
    gain1, gain2 = Float32(1 - β₁), Float32(1 - β₂)
    step = rate / Float32(1 - β₁^t)
    correction = Float32(inv(1 - β₂^t))
    epsilon = Float32(ϵ)
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
