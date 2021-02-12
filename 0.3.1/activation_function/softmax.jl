module Softmax_CEL
    function opt_func(values)
        return exp.(values.-maximum(values))/sum(exp.(values.-maximum(values)))
    end

    function func(value_matrix::Array{Float32})
        output_matrix = zeros(Float32, size(value_matrix))
        Threads.@threads for b in 1:size(value_matrix, 2)
            output_matrix[:,b] = opt_func(value_matrix[:,b])
        end
        return output_matrix
    end

    function pre_diff(inputs::Array{Float32}, position::Int64)
        outputs = opt_func(inputs)
        derivative_vector = zeros(Float32, size(inputs))
        Threads.@threads for i in 1:length(derivative_vector)
            derivative_vector[i] = i!=position ? outputs[i] : outputs[i]-1
        end
        return derivative_vector
    end

    function get_∇biases(input_matrix::Array{Float32}, propagation_units::Array{Float32})
        derivative = zeros(Float32, size(input_matrix))
        layer_size = size(input_matrix, 1)
        batch_size = size(input_matrix, 2)
        for b in 1:batch_size
            for l in 1:layer_size
                if propagation_units[l,b]!=0
                    derivative[:,b] += pre_diff(input_matrix[:,b], l)
                end
            end
        end
        return derivative
    end

    function get_name()
        return "Softmax_CEL"
    end
end

module Softmax
    function opt_func(values)
        return exp.(values.-maximum(values))/sum(exp.(values.-maximum(values)))
    end

    function func(value_matrix::Array{Float32})
        output_matrix = zeros(Float32, size(value_matrix))
        Threads.@threads for b in 1:size(value_matrix, 2)
            output_matrix[:,b] = opt_func(value_matrix[:,b])
        end
        return output_matrix
    end

    function pre_diff(inputs::Array{Float32}, position::Int64)
        outputs = opt_func(inputs)
        derivative_vector = zeros(Float32, size(inputs))
        Threads.@threads for i in 1:length(derivative_vector)
            derivative_vector[i] = i!=position ? -outputs[i]*outputs[position] : outputs[i]*(1-outputs[i])
        end
        return derivative_vector
    end

    function get_∇biases(input_matrix::Array{Float32}, propagation_units::Array{Float32})
        derivative = zeros(Float32, size(input_matrix))
        layer_size = size(input_matrix, 1)
        batch_size = size(input_matrix, 2)
        for b in 1:batch_size
            for l in 1:layer_size
                if propagation_units[l,b]!=0
                    derivative[:,b] += pre_diff(input_matrix[:,b], l)
                end
            end
        end
        return derivative.*propagation_units
    end

    function get_name()
        return "Softmax"
    end
end
