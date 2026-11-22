module Softmax
    using NNlib: softmax!
    using ..DianoiaML: accumulation_type

    function func!(output::Array{T}, values::Array{T}) where {T<:AbstractFloat}
        if T <: Union{Float32,Float64}
            return softmax!(output, values; dims=1)
        end
        # Keep the exponential and normalization in the accumulation type.
        A = accumulation_type(T)
        classes = size(values, 1)
        for first in 1:classes:length(values)
            last = first + classes - 1
            column = view(values, first:last)
            peak = A(maximum(column))
            total = sum(x -> exp(A(x) - peak), column)
            @inbounds for i in first:last
                output[i] = exp(A(values[i]) - peak) / total
            end
        end
        return output
    end

    function get_∇biases!(gradient::Array{T}, values::Array{T}, delta::Array{T}) where {T<:AbstractFloat}
        size(gradient) == size(values) == size(delta) || throw(DimensionMismatch("softmax gradient shapes differ"))
        func!(gradient, values)
        classes = size(values, 1)
        for first in 1:classes:length(values)
            last = first + classes - 1
            A = accumulation_type(T)
            projection = zero(A)
            @inbounds @simd for i in first:last
                projection += A(gradient[i]) * A(delta[i])
            end
            @inbounds @simd for i in first:last
                gradient[i] *= delta[i] - projection
            end
        end
    end

    get_name() = "Softmax"
end
