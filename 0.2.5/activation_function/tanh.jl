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

    function diff(input_matrix::Array{Float32})
        derivative = zeros(Float32, size(input_matrix))
        Threads.@threads for i in eachindex(input_matrix)
            derivative[i] = opt_diff(input_matrix[i])
        end
        return derivative
    end

    function get_name()
        return "tanH"
    end
end
