module Sigmoid
    using LoopVectorization

    function func!(output_matrix::Array{Float32}, value_matrix::Array{Float32})
        @avxt for i in eachindex(value_matrix)
            value_matrix[i] = ifelse(value_matrix[i]>3.0f38, 3.0f38, value_matrix[i])
            value_matrix[i] = ifelse(value_matrix[i]<-3.0f38, -3.0f38, value_matrix[i])
            output_matrix[i] = ifelse(value_matrix[i]>=0, 1/(1+exp(-value_matrix[i])), exp(value_matrix[i])/(1+exp(value_matrix[i])))
        end
    end

    function get_∇biases!(∇biases::Array{Float32}, value_matrix::Array{Float32}, δ::Array{Float32})
        @avxt for i in eachindex(value_matrix)
            ∇biases[i] = exp(-value_matrix[i])/(1+exp(-value_matrix[i]))^2*δ[i]
        end
    end

    function get_name()
        return "Sigmoid"
    end
end

# Source: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
