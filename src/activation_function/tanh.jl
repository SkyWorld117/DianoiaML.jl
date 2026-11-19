module tanH
    function func!(output::Array{Float32}, values::Array{Float32})
        output .= tanh.(values)
    end

    function get_∇biases!(gradient::Array{Float32}, values::Array{Float32}, delta::Array{Float32})
        gradient .= (1f0 .- tanh.(values).^2) .* delta
    end

    get_name() = "tanH"
end
