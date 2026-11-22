module GELU
    using ..DianoiaML: accumulation_type

    # Tanh approximation to the Gaussian error linear unit.
    gelu_tanh(x, c, k) = tanh(c * (x + k * x^3))

    function gelu(x, c, k)
        x == -Inf && return zero(x)
        return x * ((one(x) + gelu_tanh(x, c, k)) / 2)
    end

    function derivative(x, c, k)
        t = gelu_tanh(x, c, k)
        abs(t) == one(t) && return (one(x) + t) / 2
        return (one(x) + t + x * (one(t) - t^2) * c * (one(x) + 3 * k * x^2)) / 2
    end

    function func!(output::Array{T}, values::Array{T}) where {T<:AbstractFloat}
        A = accumulation_type(T)
        c, k = sqrt(A(2) / A(pi)), A(44715//1000000)
        output .= gelu.(A.(values), c, k)
    end

    function get_∇biases!(gradient::Array{T}, values::Array{T}, delta::Array{T}) where {T<:AbstractFloat}
        A = accumulation_type(T)
        c, k = sqrt(A(2) / A(pi)), A(44715//1000000)
        gradient .= derivative.(A.(values), c, k) .* A.(delta)
    end

    get_name() = "GELU"
end
