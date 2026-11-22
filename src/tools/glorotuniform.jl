module GlorotUniform
    using ..DianoiaML: accumulation_type, random_precision

    function GUMatrix(input_size::Int, layer_size::Int, shape, ::Type{T}=Float32) where {T<:AbstractFloat}
        A = accumulation_type(T)
        limit = sqrt(A(6) / A(input_size + layer_size))
        values = rand(random_precision(A), shape)
        return T.((A(2) * limit) .* values .- limit)
    end

    export GUMatrix
end
