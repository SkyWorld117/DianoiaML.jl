module Quadratic_Loss
    function func(output::Array{Float32}, target::Array{Float32})
        return (output .- target).^2
    end

    function prop!(delta::Array{Float32}, output::Array{Float32}, target::Array{Float32})
        delta .= 2f0 .* (output .- target)
    end
end
