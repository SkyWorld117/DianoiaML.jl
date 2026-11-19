module Sigmoid
    @inline function sigmoid(value)
        z = exp(-abs(value))
        return ifelse(value >= 0f0, 1f0, z) / (1f0 + z)
    end

    @inline function sigmoid_derivative(value)
        z = exp(-abs(value))
        return z / (1f0 + z)^2
    end

    function func!(output::Array{Float32}, values::Array{Float32})
        output .= sigmoid.(values)
    end

    function get_∇biases!(gradient::Array{Float32}, values::Array{Float32}, delta::Array{Float32})
        gradient .= sigmoid_derivative.(values) .* delta
    end

    get_name() = "Sigmoid"
end
