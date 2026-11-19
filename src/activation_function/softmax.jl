module Softmax
    using NNlib: softmax!

    function func!(output::Array{Float32}, values::Array{Float32})
        softmax!(output, values; dims=1)
    end

    function get_∇biases!(gradient::Array{Float32}, values::Array{Float32}, delta::Array{Float32})
        size(gradient) == size(values) == size(delta) || throw(DimensionMismatch("softmax gradient shapes differ"))
        func!(gradient, values)
        classes = size(values, 1)
        for first in 1:classes:length(values)
            last = first + classes - 1
            projection = 0f0
            @inbounds @simd for i in first:last
                projection += gradient[i] * delta[i]
            end
            @inbounds @simd for i in first:last
                gradient[i] *= delta[i] - projection
            end
        end
    end

    get_name() = "Softmax"
end

module Softmax_CEL
    using ..Softmax: func!, get_∇biases!
    get_name() = "Softmax_CEL"
end
