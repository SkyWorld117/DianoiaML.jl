module Absolute
    using LoopVectorization

    function func(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        loss = 0.0
        @avxt for i in eachindex(output_matrix)
            loss += abs(output_matrix[i]-sample_matrix[i])
        end
        return loss
    end
end
