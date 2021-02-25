module None
    using .Threads

    function func(value_matrix::Array{Float32})
        @threads for i in eachindex(value_matrix)
            if value_matrix[i] >= 3.0f38
                value_matrix[i] = 3.0f38
            elseif value_matrix[i] <= -3.0f38
                value_matrix[i] = -3.0f38
            end
        end
        output_matrix = deepcopy(value_matrix)
        return output_matrix
    end

    function get_∇biases(input_matrix::Array{Float32}, propagation_units::Array{Float32})
        return propagation_units
    end

    function get_name()
        return "None"
    end
end
