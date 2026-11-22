module tanH
    function func!(output::Array{T}, values::Array{T}) where {T<:AbstractFloat}
        output .= tanh.(values)
    end

    function get_∇biases!(gradient::Array{T}, values::Array{T}, delta::Array{T}) where {T<:AbstractFloat}
        gradient .= (one(T) .- tanh.(values).^2) .* delta
    end

    get_name() = "tanH"
end
