using .Threads

module Classification
    function func(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        loss = Threads.Atomic{Int64}(0)
        Threads.@threads for i in 1:size(output_matrix, 2)
            if findmax(output_matrix[:,i])[2]!=findmax(sample_matrix[:,i])[2]
                Threads.atomic_add!(loss, 1)
            end
        end
        return string(loss[])*"/"*string(size(output_matrix, 2))
    end
end
