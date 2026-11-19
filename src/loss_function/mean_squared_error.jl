module Mean_Squared_Error
    features(output) = Float32(ndims(output) == 1 ? length(output) : length(output) ÷ size(output, ndims(output)))

    function func(output::Array{Float32}, target::Array{Float32})
        return (output .- target).^2 ./ features(output)
    end

    function prop!(delta::Array{Float32}, output::Array{Float32}, target::Array{Float32})
        delta .= (2f0 / features(output)) .* (output .- target)
    end
end
