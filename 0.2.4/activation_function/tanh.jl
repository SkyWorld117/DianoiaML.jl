using .Threads

module tanH
    function opt_diff(value)
        return 1-tanh(value)^2
    end

    function func(value_matrix::Array{Float32})
        output_matrix = zeros(Float32, size(value_matrix))
        Threads.@threads for i in eachindex(output_matrix)
            output_matrix[i] = tanh(value_matrix[i])
        end
        return output_matrix
    end

    function diff(inputs::Array{Float32}, position::Int64)
        derivative_vector = zeros(Float32, size(inputs))
        derivative_vector[position] = opt_diff(inputs[position])
        return derivative_vector
    end

    function get_name()
        return "tanH"
    end
end
