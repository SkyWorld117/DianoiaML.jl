module Quadratic_Loss
    function func(output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        return (output .- target).^2
    end

    function prop!(delta::AbstractArray{<:AbstractFloat}, output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        delta .= eltype(delta)(2) .* (output .- target)
    end
end
