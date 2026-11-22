function oneHot(input::AbstractArray{<:Integer}, depth::Int, dict::AbstractDict;
                on_value::Real=1, off_value::Real=0, precision::Type{T}=Float32) where {T<:AbstractFloat}
    output = fill(checked_precision(T)(off_value), depth, length(input))
    for (column, label) in enumerate(input)
        output[dict[label], column] = on_value
    end
    return output
end
