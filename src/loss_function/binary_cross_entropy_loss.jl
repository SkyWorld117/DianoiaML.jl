module Binary_Cross_Entropy_Loss
    const EPSILON = 1f-8
    binary_loss(p, y) = -y * log(max(p, EPSILON)) - (1f0 - y) * log(max(1f0 - p, EPSILON))
    binary_gradient(p, y) = (1f0 - y) / max(1f0 - p, EPSILON) - y / max(p, EPSILON)

    function func(output::Array{Float32}, target::Array{Float32})
        return binary_loss.(output, target)
    end

    function prop!(delta::Array{Float32}, output::Array{Float32}, target::Array{Float32})
        delta .= binary_gradient.(output, target)
    end
end
