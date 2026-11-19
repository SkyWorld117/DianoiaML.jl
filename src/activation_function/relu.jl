module ReLU
    function func!(output::Array{Float32}, values::Array{Float32})
        output .= max.(0f0, values)
    end

    function get_∇biases!(gradient::Array{Float32}, values::Array{Float32}, delta::Array{Float32})
        gradient .= ifelse.(values .> 0f0, delta, 0f0)
    end

    get_name() = "ReLU"
end
