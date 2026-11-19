function oneHot(input::AbstractArray{<:Integer}, depth::Int, dict::AbstractDict; on_value::Float32=1f0, off_value::Float32=0f0)
    output = fill(off_value, depth, length(input))
    for (column, label) in enumerate(input)
        output[dict[label], column] = on_value
    end
    return output
end
