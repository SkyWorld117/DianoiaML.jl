module Classification
    using Polyester

    function func(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        loss = zeros(Int64, size(output_matrix)[2])
        @batch for b in axes(output_matrix, 2)
            if findmax(output_matrix[:,b])[2]!=findmax(sample_matrix[:,b])[2]
                loss[b] += 1
            end
        end
        return sum(loss)
    end
end
