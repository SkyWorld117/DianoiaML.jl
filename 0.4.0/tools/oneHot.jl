using LoopVectorization

function oneHot(input::Array{Int64}, depth::Int64, dict::Dict; on_value::Float32=1.0f0, off_value::Float32=0.0f0)
    output_data = fill(off_value, (depth, size(input, 1)))
    @avxt for i in 1:length(input)
        output_data[dict[input[i]],i] = on_value
    end
    return output_data
end
