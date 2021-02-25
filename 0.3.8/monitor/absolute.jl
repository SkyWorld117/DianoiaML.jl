module Absolute
    using .Threads

    function func(output_matrix::Array{Float32}, sample_matrix::Array{Float32})
        loss = Threads.Atomic{Float64}(0)
        @threads for i in eachindex(output_matrix)
            Threads.atomic_add!(loss, abs(output_matrix[i]-sample_matrix[i]))
        end
        return loss[]
    end
end
