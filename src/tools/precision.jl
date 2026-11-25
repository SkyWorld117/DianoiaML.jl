function checked_precision(::Type{T}) where {T<:AbstractFloat}
    isconcretetype(T) || throw(ArgumentError("precision must be a concrete floating-point type"))
    return T
end

accumulation_type(::Type{T}) where {T<:AbstractFloat} =
    isbitstype(T) && sizeof(T) < sizeof(Float32) ? Float32 : T

random_precision(::Type{T}) where {T<:Union{Float32,Float64,BigFloat}} = T
random_precision(::Type{<:AbstractFloat}) = BigFloat

precision_type(layer) = nothing

float_input(precision, input) = input
float_input(::Type{T}, input::Union{Tuple,NamedTuple}) where {T} = map(x -> float_input(T, x), input)
function float_input(::Type{T}, input::AbstractArray{<:AbstractFloat}) where {T}
    backend = device_of(input)
    if backend !== nothing
        check_gpu_precision(backend, T)
        return eltype(input) === T ? input : T.(input)
    end
    return input isa Array{T} ? input : Array{T}(input)
end

cotangent_as(prediction, delta) = delta
cotangent_as(prediction::Tuple, delta::Tuple) = map(cotangent_as, prediction, delta)
cotangent_as(prediction::NamedTuple, delta::NamedTuple) = map(cotangent_as, prediction, delta)
function cotangent_as(prediction::AbstractArray{T}, delta::AbstractArray) where {T<:AbstractFloat}
    return float_input(T, batch_on(device_of(prediction), delta))
end

function probability_floor(::Type{T}) where {T<:AbstractFloat}
    A = accumulation_type(T)
    return T === Float16 ? max(A(1//100000000), inv(A(floatmax(T)))) : A(1//100000000)
end
probability_floor(::Type{BigFloat}) = eps(BigFloat)
