module Absolute_Loss
    function func(output::Array{Float32}, target::Array{Float32})
        return abs.(output .- target)
    end

    function prop!(delta::Array{Float32}, output::Array{Float32}, target::Array{Float32})
        delta .= sign.(output .- target)
    end
end
