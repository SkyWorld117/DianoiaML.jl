module None
    using .Threads, LoopVectorization

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

    function get_∇biases!(∇biases::Array{Float32}, input_matrix::Array{Float32}, propagation_units::Array{Float32})
        @avx for i in axes(∇biases, 1), j in axes(∇biases, 2)
            ∇biases[i,j] = propagation_units[i,j]
        end
    end

    function get_name()
        return "None"
    end
end
