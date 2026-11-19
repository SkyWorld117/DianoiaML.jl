module Categorical_Cross_Entropy_Loss
    function func(output::Array{Float32}, target::Array{Float32})
        return -target .* log.(max.(output, 1f-8))
    end

    function prop!(delta::Array{Float32}, output::Array{Float32}, target::Array{Float32})
        delta .= -target ./ max.(output, 1f-8)
    end
end
