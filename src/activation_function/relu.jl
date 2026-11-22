module ReLU
    function func!(output::Array{T}, values::Array{T}) where {T<:AbstractFloat}
        output .= max.(zero(T), values)
    end

    function get_∇biases!(gradient::Array{T}, values::Array{T}, delta::Array{T}) where {T<:AbstractFloat}
        gradient .= ifelse.(values .> zero(T), delta, zero(T))
    end

    get_name() = "ReLU"
end
