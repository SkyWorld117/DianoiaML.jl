module Absolute_Loss
    function func(output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        return abs.(output .- target)
    end

    function prop!(delta::AbstractArray{<:AbstractFloat}, output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        delta .= sign.(output .- target)
    end
end
