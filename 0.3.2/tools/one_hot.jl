using .Threads

function One_Hot(index::Array{Int64}, depth::Int64, dict::Dict; on_value::Float32=1.0f0, off_value::Float32=0.0f0)
    output_data = fill(off_value, (depth, size(index, 1)))
    @threads for i in 1:length(index)
        output_data[dict[index[i]],i] = on_value
    end
    return output_data
end
