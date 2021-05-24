module Absolute_Loss
    using LoopVectorization

    function func(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        loss_matrix = zeros(Float32, size(output_matrix))
        @avxt for i in axes(output_matrix, 1), j in axes(output_matrix, 2)
            loss_matrix[i,j] = abs(output_matrix[i,j]-sample_matrix[i,j])
        end
        return loss_matrix
    end

    function prop!(δ::Array{Float32}, output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        @avxt for i in eachindex(δ)
            δ[i] = ifelse(output_matrix[i]-sample_matrix[i]>=0, 1, -1)
        end
    end
end
