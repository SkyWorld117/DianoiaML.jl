module None
    function func!(output::Array{T}, values::Array{T}) where {T<:AbstractFloat}
        copyto!(output, values)
    end

    function get_∇biases!(gradient::Array{T}, values::Array{T}, delta::Array{T}) where {T<:AbstractFloat}
        copyto!(gradient, delta)
    end

    get_name() = "None"
end
