module Softmax_CEL
    using LoopVectorization

    function opt_func(values)
        return @avxt exp.(values.-maximum(values))/sum(exp.(values.-maximum(values)))
    end

    function func!(output_matrix::Array{Float32}, value_matrix::Array{Float32})
        @avxt for i in axes(value_matrix, 1), j in axes(value_matrix, 2)
            value_matrix[i,j] = ifelse(value_matrix[i,j]>3.0f38, 3.0f38, value_matrix[i,j])
            value_matrix[i,j] = ifelse(value_matrix[i,j]<-3.0f38, -3.0f38, value_matrix[i,j])
        end
        for b in axes(value_matrix, 2)
            output_matrix[:,b] = opt_func(value_matrix[:,b])
        end
    end

    function pre_diff(inputs::Array{Float32}, position::Int64)
        outputs = opt_func(inputs)
        derivative_vector = zeros(Float32, size(inputs))
        @avxt for i in 1:length(derivative_vector)
            derivative_vector[i] = ifelse(i!=position, outputs[i], outputs[i]-1)
        end
        return derivative_vector
    end

    function get_∇biases!(∇biases::Array{Float32}, input_matrix::Array{Float32}, δ::Array{Float32})
        @avxt for i in axes(∇biases, 1), j in axes(∇biases, 2)
            ∇biases[i,j] = 0.0f0
        end

        for b in axes(input_matrix, 2)
            for l in axes(input_matrix, 1)
                if δ[l,b]!=0
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
    using LoopVectorization

    function opt_func(values)
        return @avxt exp.(values.-maximum(values))/sum(exp.(values.-maximum(values)))
    end

    function func!(output_matrix::Array{Float32}, value_matrix::Array{Float32})
        @avxt for i in axes(value_matrix, 1), j in axes(value_matrix, 2)
            value_matrix[i,j] = ifelse(value_matrix[i,j]>3.0f38, 3.0f38, value_matrix[i,j])
            value_matrix[i,j] = ifelse(value_matrix[i,j]<-3.0f38, -3.0f38, value_matrix[i,j])
        end
        for b in axes(value_matrix, 2)
            output_matrix[:,b] = opt_func(value_matrix[:,b])
        end
    end

    function pre_diff(inputs::Array{Float32}, position::Int64)
        outputs = opt_func(inputs)
        derivative_vector = zeros(Float32, size(inputs))
        @avxt for i in 1:length(derivative_vector)
            derivative_vector[i] = ifelse(i!=position, -outputs[i]*outputs[position], outputs[i]*(1-outputs[i]))
        end
        return derivative_vector
    end

    function get_∇biases!(∇biases::Array{Float32}, input_matrix::Array{Float32}, δ::Array{Float32})
        @avxt for i in axes(∇biases, 1), j in axes(∇biases, 2)
            ∇biases[i,j] = 0.0f0
        end

        for b in axes(input_matrix, 2)
            for l in axes(input_matrix, 1)
                if δ[l,b]!=0
                    ∇biases[:,b] += pre_diff(input_matrix[:,b], l)
                end
            end
        end

        @avxt for i in axes(∇biases, 1), j in axes(∇biases, 2)
            ∇biases[i,j] *= δ[i,j]
        end
    end

    function get_name()
        return "Softmax"
    end
end
