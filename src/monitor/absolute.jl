module Absolute
    function func(output::Array{Float32}, target::Array{Float32})
        size(output) == size(target) || throw(DimensionMismatch("prediction and target shapes differ"))
        loss = 0.0
        @inbounds @simd for i in eachindex(output, target)
            loss += abs(output[i] - target[i])
        end
        return loss
    end
end
