module None
    function func(value_matrix::Array{Float32})
        output_matrix = deepcopy(value_matrix)
        return output_matrix
    end

    function diff(input_matrix::Array{Float32}, propagation_units::Array{Float32})
        return propagation_units
    end

    function get_name()
        return "None"
    end
end
