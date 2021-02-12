module Mean_Squared_Error
    function opt_func(output::Float32, sample::Float32)
        return (output-sample)^2
    end

    function opt_pu(output::Float32, sample::Float32)
        return 2*(output-sample)
    end

    function func(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        layer_size = size(output_matrix, 1)
        loss_matrix = zeros(Float32, size(output_matrix))
        Threads.@threads for i in eachindex(loss_matrix)
            loss_matrix[i] = opt_func(output_matrix[i], sample_matrix[i])/layer_size
        end
        return loss_matrix
    end

    function prop(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        layer_size = size(output_matrix, 1)
        propagation_units = zeros(Float32, size(output_matrix))
        Threads.@threads for i in eachindex(propagation_units)
            propagation_units[i] = opt_pu(output_matrix[i], sample_matrix[i])/layer_size
        end
        return propagation_units
    end
end
