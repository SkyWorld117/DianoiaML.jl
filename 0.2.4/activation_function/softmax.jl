using .Threads

module Softmax
    function opt_func(values)
        return exp.(values.-maximum(values))/sum(exp.(values.-maximum(values)))
    end

    function func(value_matrix::Array{Float32})
        output_matrix = zeros(Float32, size(value_matrix))
        Threads.@threads for i in 1:size(value_matrix, 2)
            output_matrix[:,i] = opt_func(value_matrix[:,i])
        end
        return output_matrix
    end

    function diff(inputs::Array{Float32}, position::Int64)
        outputs = opt_func(inputs)
        derivative_vector = zeros(Float32, size(inputs))
        for i in 1:length(derivative_vector)
            derivative_vector[i] = i!=position ? -outputs[i]*outputs[position] : outputs[i]*(1-outputs[i])
        end
        return derivative_vector
    end

    function get_name()
        return "Softmax"
    end
end
