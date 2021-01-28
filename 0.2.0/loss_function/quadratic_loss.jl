using .Threads

module Quadratic_Loss
    function opt_func(output::Float32, sample::Float32)
        return (output-sample)^2
    end

    function opt_pu(output::Float32, sample::Float32)
        return 2*(output-sample)
    end

    function func(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        loss_matrix = zeros(Float32, size(output_matrix))
        Threads.@threads for i in eachindex(loss_matrix)
            loss_matrix[i] = opt_func(output_matrix[i], sample_matrix[i])
        end
        return loss_matrix
    end

    function prop(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        propagation_units = zeros(Float32, size(output_matrix))
        Threads.@threads for i in eachindex(propagation_units)
            propagation_units[i] = opt_pu(output_matrix[i], sample_matrix[i])
        end
        return propagation_units
    end
end
