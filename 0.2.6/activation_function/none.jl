module None
    function func(value_matrix::Array{Float32})
        output_matrix = deepcopy(value_matrix)
        return output_matrix
    end

    function diff(input_matrix::Array{Float32})
        return ones(Float32, size(input_matrix))
    end

    function get_name()
        return "None"
    end
end
