module Categorical_Cross_Entropy_Loss
    using ..DianoiaML: probability_floor
    function func(output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        return -target .* log.(max.(output, probability_floor(eltype(output))))
    end

    function prop!(delta::AbstractArray{<:AbstractFloat}, output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        delta .= -target ./ max.(output, probability_floor(eltype(output)))
    end
end
