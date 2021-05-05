module Quadratic_Loss
    using LoopVectorization

    function func(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        loss_matrix = zeros(Float32, size(output_matrix))
        @avxt for i in axes(loss_matrix, 1), j in axes(loss_matrix, 2)
            loss_matrix[i,j] = (output_matrix[i,j]-sample_matrix[i,j])^2
        end
        return loss_matrix
    end

    function prop!(δ::Array{Float32}, output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        @avxt for i in eachindex(i)
            δ[i] = 2*(output_matrix[i]-sample_matrix[i])
        end
    end
end
