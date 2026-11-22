module Sigmoid
    @inline function sigmoid(value)
        z = exp(-abs(value))
        unit = one(z)
        return ifelse(value >= zero(value), unit, z) / (unit + z)
    end

    @inline function sigmoid_derivative(value)
        z = exp(-abs(value))
        unit = one(z)
        return z / (unit + z)^2
    end

    function func!(output::Array{T}, values::Array{T}) where {T<:AbstractFloat}
        output .= sigmoid.(values)
    end

    function get_∇biases!(gradient::Array{T}, values::Array{T}, delta::Array{T}) where {T<:AbstractFloat}
        gradient .= sigmoid_derivative.(values) .* delta
    end

    get_name() = "Sigmoid"
end
