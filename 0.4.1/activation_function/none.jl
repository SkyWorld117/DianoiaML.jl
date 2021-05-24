module None
    using LoopVectorization

    function func!(output_matrix::Array{Float32}, value_matrix::Array{Float32})
        @avxt for i in eachindex(value_matrix)
            value_matrix[i] = ifelse(value_matrix[i]>3.0f38, 3.0f38, value_matrix[i])
            value_matrix[i] = ifelse(value_matrix[i]<-3.0f38, -3.0f38, value_matrix[i])
            output_matrix[i] = value_matrix[i]
        end
    end

    function get_∇biases!(∇biases::Array{Float32}, value_matrix::Array{Float32}, δ::Array{Float32})
        @avxt for i in eachindex(∇biases)
            ∇biases[i] = δ[i]
        end
    end

    function get_name()
        return "None"
    end
end
