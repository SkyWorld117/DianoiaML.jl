module Mean_Squared_Error
    using LoopVectorization

    function func(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        layer_size = size(output_matrix, 1)
        loss_matrix = zeros(Float32, size(output_matrix))
        @avxt for i in axes(loss_matrix, 1), j in axes(loss_matrix, 2)
            loss_matrix[i,j] = (output_matrix[i,j]-sample_matrix[i,j])^2/layer_size
        end
        return loss_matrix
    end

    function prop!(δ::Array{Float32}, output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        layer_size = size(output_matrix, 1)
        @avxt for i in eachindex(δ)
            δ[i] = 2*(output_matrix[i]-sample_matrix[i])/layer_size
        end
    end
end
