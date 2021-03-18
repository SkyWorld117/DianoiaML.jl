module Softmax_CEL
    using .Threads, LoopVectorization

    function opt_func(values)
        return exp.(values.-maximum(values))/sum(exp.(values.-maximum(values)))
    end

    function func(value_matrix::Array{Float32})
        output_matrix = zeros(Float32, size(value_matrix))
        @threads for i in eachindex(value_matrix)
            if value_matrix[i] >= 3.0f38
                value_matrix[i] = 3.0f38
            elseif value_matrix[i] <= -3.0f38
                value_matrix[i] = -3.0f38
            end
        end
        @threads for b in axes(value_matrix, 2)
            output_matrix[:,b] = opt_func(value_matrix[:,b])
        end
        return output_matrix
    end

    function pre_diff(inputs::Array{Float32}, position::Int64)
        outputs = opt_func(inputs)
        derivative_vector = zeros(Float32, size(inputs))
        @threads for i in 1:length(derivative_vector)
            derivative_vector[i] = i!=position ? outputs[i] : outputs[i]-1
        end
        return derivative_vector
    end

    function get_∇biases!(∇biases::Array{Float32}, input_matrix::Array{Float32}, propagation_units::Array{Float32})
        @avx for i in axes(∇biases, 1), j in axes(∇biases, 2)
            ∇biases[i,j] = 0.0f0
        end

        for b in axes(input_matrix, 2)
            for l in axes(input_matrix, 1)
                if propagation_units[l,b]!=0
                    ∇biases[:,b] += pre_diff(input_matrix[:,b], l)
                end
            end
        end
    end

    function get_name()
        return "Softmax_CEL"
    end
end

module Softmax
    using .Threads, LoopVectorization

    function opt_func(values)
        return exp.(values.-maximum(values))/sum(exp.(values.-maximum(values)))
    end

    function func(value_matrix::Array{Float32})
        output_matrix = zeros(Float32, size(value_matrix))
        @threads for i in eachindex(value_matrix)
            if value_matrix[i] >= 3.0f38
                value_matrix[i] = 3.0f38
            elseif value_matrix[i] <= -3.0f38
                value_matrix[i] = -3.0f38
            end
        end
        @threads for b in axes(value_matrix, 2)
            output_matrix[:,b] = opt_func(value_matrix[:,b])
        end
        return output_matrix
    end

    function pre_diff(inputs::Array{Float32}, position::Int64)
        outputs = opt_func(inputs)
        derivative_vector = zeros(Float32, size(inputs))
        @threads for i in 1:length(derivative_vector)
            derivative_vector[i] = i!=position ? -outputs[i]*outputs[position] : outputs[i]*(1-outputs[i])
        end
        return derivative_vector
    end

    function get_∇biases!(∇biases::Array{Float32}, input_matrix::Array{Float32}, propagation_units::Array{Float32})
        @avx for i in axes(∇biases, 1), j in axes(∇biases, 2)
            ∇biases[i,j] = 0.0f0
        end

        for b in axes(input_matrix, 2)
            for l in axes(input_matrix, 1)
                if propagation_units[l,b]!=0
                    ∇biases[:,b] += pre_diff(input_matrix[:,b], l)
                end
            end
        end

        @avx for i in axes(∇biases, 1), j in axes(∇biases, 2)
            ∇biases[i,j] *= propagation_units[i,j]
        end
    end

    function get_name()
        return "Softmax"
    end
end
