module Binary_Cross_Entropy_Loss
    using ..DianoiaML: probability_floor

    binary_loss(p, y, ε) = -y * log(max(p, ε)) - (one(ε) - y) * log(max(one(ε) - p, ε))
    binary_gradient(p, y, ε) = (one(ε) - y) / max(one(ε) - p, ε) - y / max(p, ε)

    function func(output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        return binary_loss.(output, target, probability_floor(eltype(output)))
    end

    function prop!(delta::AbstractArray{<:AbstractFloat}, output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        delta .= binary_gradient.(output, target, probability_floor(eltype(output)))
    end
end
