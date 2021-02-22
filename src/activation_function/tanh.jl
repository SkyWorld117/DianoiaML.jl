module tanH
    using .Threads

    function opt_diff(value)
        return 1-tanh(value)^2
    end

    function func(value_matrix::Array{Float32})
        output_matrix = zeros(Float32, size(value_matrix))
        @threads for i in eachindex(value_matrix)
            if value_matrix[i] >= 3.0f38
                value_matrix[i] = 3.0f38
            elseif value_matrix[i] <= -3.0f38
                value_matrix[i] = -3.0f38
            end
            output_matrix[i] = tanh(value_matrix[i])
        end
        return output_matrix
    end

    function get_∇biases(input_matrix::Array{Float32}, propagation_units::Array{Float32})
        derivative = zeros(Float32, size(input_matrix))
        @threads for i in eachindex(input_matrix)
            derivative[i] = opt_diff(input_matrix[i])
        end
        return derivative.*propagation_units
    end

    function get_name()
        return "tanH"
    end
end
