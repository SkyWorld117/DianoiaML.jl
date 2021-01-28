using .Threads

module None
    function func(value_matrix::Array{Float32})
        output_matrix = deepcopy(value_matrix)
        return output_matrix
    end

    function diff(inputs::Array{Float32}, position::Int64)
        derivative_vector = zeros(Float32, size(inputs))
        derivative_vector[position] = 1.0f0
        return derivative_vector
    end

    function get_name()
        return "None"
    end
end
