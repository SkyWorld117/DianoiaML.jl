module Absolute
    using ..DianoiaML: accumulation_type

    function func(output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        size(output) == size(target) || throw(DimensionMismatch("prediction and target shapes differ"))
        loss = zero(promote_type(Float64, accumulation_type(eltype(output)), accumulation_type(eltype(target))))
        @inbounds @simd for i in eachindex(output, target)
            loss += abs(output[i] - target[i])
        end
        return loss
    end
end
