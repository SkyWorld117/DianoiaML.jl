module Mean_Squared_Error
    using ..DianoiaML: accumulation_type
    features(output) = accumulation_type(eltype(output))(ndims(output) == 1 ? length(output) : length(output) ÷ size(output, ndims(output)))

    function func(output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        divisor = features(output)
        A = typeof(divisor)
        return ((x, y) -> abs2(A(x) - A(y)) / divisor).(output, target)
    end

    function prop!(delta::AbstractArray{<:AbstractFloat}, output::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:AbstractFloat})
        A = accumulation_type(eltype(delta))
        scale = A(2) / features(output)
        delta .= ((x, y) -> scale * (A(x) - A(y))).(output, target)
    end
end
