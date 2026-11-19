module None
    function func!(output::Array{Float32}, values::Array{Float32})
        copyto!(output, values)
    end

    function get_∇biases!(gradient::Array{Float32}, values::Array{Float32}, delta::Array{Float32})
        copyto!(gradient, delta)
    end

    get_name() = "None"
end
