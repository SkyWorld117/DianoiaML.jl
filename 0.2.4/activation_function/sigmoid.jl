using .Threads

module Sigmoid
    function opt_func(value)
        return value>=0 ? 1/(1+exp(-value)) : exp(value)/(1+exp(value))
    end

    function opt_diff(value)
        return exp(-value)/(1+exp(-value))^2
    end

    function func(value_matrix::Array{Float32})
        output_matrix = zeros(Float32, size(value_matrix))
        Threads.@threads for i in eachindex(output_matrix)
            output_matrix[i] = opt_func(value_matrix[i])
        end
        return output_matrix
    end

    function diff(inputs::Array{Float32}, position::Int64)
        derivative_vector = zeros(Float32, size(inputs))
        derivative_vector[position] = opt_diff(inputs[position])
        return derivative_vector
    end

    function get_name()
        return "Sigmoid"
    end
end

# Source: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
